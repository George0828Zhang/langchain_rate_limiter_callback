import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, ChatGeneration

logger = logging.getLogger(__name__)


class RateLimiterCallback(BaseCallbackHandler):
    """A standalone LangChain Callback Handler that enforces RPM and TPM rate limits
    directly before and after LLM execution hooks.
    """

    def __init__(
        self,
        *,
        get_token_count: Callable[[List[BaseMessage]], int],
        requests_per_minute: float = 60.0,
        tokens_per_minute: float = 50000.0,
        estimate_generation_token_count: Union[int, Callable[[List[BaseMessage]], int]] = 100,
        check_every_n_seconds: float = 0.05,
        verbose: bool = False,
        custom_logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()

        self.get_token_count = get_token_count
        self.rpm_rate = requests_per_minute / 60.0
        self.tpm_rate = tokens_per_minute / 60.0

        self.max_rpm_bucket = requests_per_minute
        self.max_tpm_bucket = tokens_per_minute

        self.available_requests = requests_per_minute
        self.available_tokens = tokens_per_minute

        self.check_every_n_seconds = check_every_n_seconds
        self.estimate_gen = estimate_generation_token_count

        self.verbose = verbose
        self.log = custom_logger if custom_logger is not None else logger

        self._lock = threading.Lock()
        self.last_refill: Optional[float] = None
        self._run_token_reservations: Dict[str, int] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError  # triggers fallback to on_llm_start

    def _refill_buckets(self, now: float) -> None:
        if self.last_refill is None:
            self.last_refill = now
            return

        elapsed = now - self.last_refill
        if elapsed > 0:
            self.available_requests += elapsed * self.rpm_rate
            self.available_requests = min(self.available_requests, self.max_rpm_bucket)

            self.available_tokens += elapsed * self.tpm_rate
            self.available_tokens = min(self.available_tokens, self.max_tpm_bucket)

            self.last_refill = now

    def _get_estimated_tokens(self, messages: List[BaseMessage]) -> int:
        if callable(self.estimate_gen):
            return self.estimate_gen(messages)
        return self.estimate_gen

    def _consume(self, messages: List[BaseMessage]) -> Tuple[bool, Optional[str], int]:
        with self._lock:
            now = time.monotonic()
            self._refill_buckets(now)

            # Now safely calculates prompt tokens for both string prompts and structural messages
            prompt_tokens = self.get_token_count(messages) if messages else 0
            estimated_gen = self._get_estimated_tokens(messages)
            required_tokens = prompt_tokens + estimated_gen

            rpm_satisfied = self.available_requests >= 1
            tpm_satisfied = self.available_tokens > 0  # Optimistic check

            if rpm_satisfied and tpm_satisfied:
                self.available_requests -= 1
                self.available_tokens -= required_tokens
                return True, None, required_tokens

            if not rpm_satisfied and not tpm_satisfied:
                reason = "both RPM and TPM limits reached"
            elif not rpm_satisfied:
                reason = "RPM limit reached"
            else:
                reason = f"TPM limit reached (Current balance: {self.available_tokens:.1f}, requested estimate: {required_tokens})"

            return False, reason, 0

    def _adjust_tokens(self, reserved_tokens: int, actual_tokens: int) -> None:
        with self._lock:
            difference = reserved_tokens - actual_tokens
            self.available_tokens += difference
            self.available_tokens = min(self.available_tokens, self.max_tpm_bucket)

            if self.verbose:
                self.log.debug(
                    f"[RateLimiterCallback] Token adjustment: Reserved {reserved_tokens}, "
                    f"Actual {actual_tokens}. Refunded/Deducted: {difference}. "
                    f"Current available tokens: {self.available_tokens:.2f}"
                )

    def _resolve_input_messages(self, prompts: List[str], kwargs: Any) -> List[BaseMessage]:
        """Resolves structural messages from execution arguments, falling back to

        wrapping raw string prompts into standard HumanMessages if needed.
        """
        # 1. Try to extract structured chat messages if available
        msg_lists = kwargs.get("messages")
        if msg_lists and isinstance(msg_lists, list) and len(msg_lists) > 0:
            return msg_lists[0]

        # 2. Fallback to raw text prompts list passed to LLM models
        if prompts:
            return [HumanMessage(content=p) for p in prompts]

        return []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: Any, **kwargs: Any
    ) -> None:
        messages = self._resolve_input_messages(prompts, kwargs)
        consumed, reason, required_tokens = self._consume(messages)
        has_logged_waiting = False

        while not consumed:
            if self.verbose and not has_logged_waiting:
                self.log.warning(
                    f"[RateLimiterCallback] Target limits reached. Request is waiting because {reason}."
                )
                has_logged_waiting = True
            time.sleep(self.check_every_n_seconds)
            consumed, reason, required_tokens = self._consume(messages)

        if self.verbose and has_logged_waiting:
            self.log.info("[RateLimiterCallback] Capacity acquired! Resuming request.")

        self._run_token_reservations[str(run_id)] = required_tokens

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        actual_usage = 0
        if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
            metadata = generation.message.usage_metadata
            if metadata:
                actual_usage = metadata.get("total_tokens", 0)

        reserved = self._run_token_reservations.pop(str(run_id), 0)

        if reserved > 0 and actual_usage > 0:
            self._adjust_tokens(reserved, actual_usage)


class AsyncRateLimiterCallback(AsyncCallbackHandler):
    """A standalone LangChain Async Callback Handler that enforces RPM and TPM rate limits
    directly before and after LLM execution hooks.
    """

    def __init__(
        self,
        *,
        get_token_count: Callable[[List[BaseMessage]], int],
        requests_per_minute: float = 60.0,
        tokens_per_minute: float = 50000.0,
        estimate_generation_token_count: Union[int, Callable[[List[BaseMessage]], int]] = 100,
        check_every_n_seconds: float = 0.05,
        verbose: bool = False,
        custom_logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()

        self.get_token_count = get_token_count
        self.rpm_rate = requests_per_minute / 60.0
        self.tpm_rate = tokens_per_minute / 60.0

        self.max_rpm_bucket = requests_per_minute
        self.max_tpm_bucket = tokens_per_minute

        self.available_requests = requests_per_minute
        self.available_tokens = tokens_per_minute

        self.check_every_n_seconds = check_every_n_seconds
        self.estimate_gen = estimate_generation_token_count

        self.verbose = verbose
        self.log = custom_logger if custom_logger is not None else logger

        self._lock = threading.Lock()
        self.last_refill: Optional[float] = None
        self._run_token_reservations: Dict[str, int] = {}

    def _refill_buckets(self, now: float) -> None:
        if self.last_refill is None:
            self.last_refill = now
            return

        elapsed = now - self.last_refill
        if elapsed > 0:
            self.available_requests += elapsed * self.rpm_rate
            self.available_requests = min(self.available_requests, self.max_rpm_bucket)

            self.available_tokens += elapsed * self.tpm_rate
            self.available_tokens = min(self.available_tokens, self.max_tpm_bucket)

            self.last_refill = now

    def _get_estimated_tokens(self, messages: List[BaseMessage]) -> int:
        if callable(self.estimate_gen):
            return self.estimate_gen(messages)
        return self.estimate_gen

    def _consume(self, messages: List[BaseMessage]) -> Tuple[bool, Optional[str], int]:
        with self._lock:
            now = time.monotonic()
            self._refill_buckets(now)

            # Now safely calculates prompt tokens for both string prompts and structural messages
            prompt_tokens = self.get_token_count(messages) if messages else 0
            estimated_gen = self._get_estimated_tokens(messages)
            required_tokens = prompt_tokens + estimated_gen

            rpm_satisfied = self.available_requests >= 1
            tpm_satisfied = self.available_tokens > 0  # Optimistic check

            if rpm_satisfied and tpm_satisfied:
                self.available_requests -= 1
                self.available_tokens -= required_tokens
                return True, None, required_tokens

            if not rpm_satisfied and not tpm_satisfied:
                reason = "both RPM and TPM limits reached"
            elif not rpm_satisfied:
                reason = "RPM limit reached"
            else:
                reason = f"TPM limit reached (Current balance: {self.available_tokens:.1f}, requested estimate: {required_tokens})"

            return False, reason, 0

    def _adjust_tokens(self, reserved_tokens: int, actual_tokens: int) -> None:
        with self._lock:
            difference = reserved_tokens - actual_tokens
            self.available_tokens += difference
            self.available_tokens = min(self.available_tokens, self.max_tpm_bucket)

            if self.verbose:
                self.log.debug(
                    f"[AsyncRateLimiterCallback] Token adjustment: Reserved {reserved_tokens}, "
                    f"Actual {actual_tokens}. Refunded/Deducted: {difference}. "
                    f"Current available tokens: {self.available_tokens:.2f}"
                )

    def _resolve_input_messages(self, prompts: List[str], kwargs: Any) -> List[BaseMessage]:
        # 1. Try to extract structured chat messages if available
        msg_lists = kwargs.get("messages")
        if msg_lists and isinstance(msg_lists, list) and len(msg_lists) > 0:
            return msg_lists[0]

        # 2. Fallback to raw text prompts list passed to LLM models
        if prompts:
            return [HumanMessage(content=p) for p in prompts]

        return []

    async def on_llm_start_async(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: Any, **kwargs: Any
    ) -> None:
        messages = self._resolve_input_messages(prompts, kwargs)
        consumed, reason, required_tokens = self._consume(messages)
        has_logged_waiting = False

        while not consumed:
            if self.verbose and not has_logged_waiting:
                self.log.warning(
                    f"[AsyncRateLimiterCallback] Target limits reached. Request (async) is waiting because {reason}."
                )
                has_logged_waiting = True
            await asyncio.sleep(self.check_every_n_seconds)
            consumed, reason, required_tokens = self._consume(messages)

        if self.verbose and has_logged_waiting:
            self.log.info("[AsyncRateLimiterCallback] Capacity acquired! Resuming async request.")

        self._run_token_reservations[str(run_id)] = required_tokens

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        actual_usage = 0
        if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
            metadata = generation.message.usage_metadata
            if metadata:
                actual_usage = metadata.get("total_tokens", 0)

        reserved = self._run_token_reservations.pop(str(run_id), 0)

        if reserved > 0 and actual_usage > 0:
            self._adjust_tokens(reserved, actual_usage)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError  # triggers fallback to on_llm_start
