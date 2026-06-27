# LangChain Rate Limiter Callback

A LangChain Callback Handler that enforces RPM (Requests Per Minute) and TPM (Tokens Per Minute) rate limits directly before and after LLM execution hooks.

## Features

- **RPM Limiting**: Enforce limits on the number of requests sent to the LLM.
- **TPM Limiting**: Enforce limits on the number of tokens used (based on estimates and actual usage).
- **Token Adjustment**: Dynamically adjusts the available token balance based on actual usage reported by the LLM.
- **Wait Logic**: Automatically pauses requests that exceed limits until capacity becomes available.
- **Verbose Logging**: Detailed logging of rate limit status, waiting reasons, and token adjustments.

## Installation

Install the package using pip from the current directory:

```bash
pip install -e .
```

## Usage

### Basic Usage (Shared Callback Pattern)

The `LangChainRateLimiterCallback` is **stateful** and must be **shared across all LLM instances** that should be jointly rate-limited. The callback maintains internal counters for RPM and TPM; if multiple LLM instances use separate callback instances, the rate limiting control fails because each instance counts its own requests independently.

```python
from langchain_rate_limiter_callback import LangChainRateLimiterCallback
from langchain_openai import ChatOpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define token counting function
def get_token_count(messages):
    return sum(len(m.content) for m in messages if hasattr(m, 'content') and m.content)

# Initialize the rate limiter ONCE (stateful, shared across instances)
rate_limiter = LangChainRateLimiterCallback(
    get_token_count=get_token_count,
    requests_per_minute=60,      # Total RPM across ALL LLM instances
    tokens_per_minute=50000,     # Total TPM across ALL LLM instances
    verbose=True
)

# Create MULTIPLE LLM instances SHARING the same callback instance
llm1 = ChatOpenAI(model="gpt-4", callbacks=[rate_limiter])
llm2 = ChatOpenAI(model="gpt-4", callbacks=[rate_limiter])
llm3 = ChatOpenAI(model="gpt-4", callbacks=[rate_limiter])

# Both llm1, llm2 and llm3 share the same RPM/TPM budget
response1 = llm1.invoke("Tell me a joke")
response2 = llm2.invoke("Explain quantum computing")
response3 = llm3.invoke("Write a poem")

# Incorrect Pattern (FAILS rate limiting): Each LLM instance has its own callback
# This effectively multiplies your RPM/TPM limits (rate limiting broken!)
#
# from langchain_rate_limiter_callback import LangChainRateLimiterCallback
# llm1 = ChatOpenAI(model="gpt-4", callbacks=[LangChainRateLimiterCallback(
#     get_token_count=get_token_count, requests_per_minute=60, tokens_per_minute=50000
# )])
# llm2 = ChatOpenAI(model="gpt-4", callbacks=[LangChainRateLimiterCallback(
#     get_token_count=get_token_count, requests_per_minute=60, tokens_per_minute=50000
# )])
```

**Thread Safety:** The callback is designed to be used with multiple threads and processes. All LLM instances (even across different threads) that is using the same LLM endpoint should share the same callback instance for proper global rate limiting.

## Rate Limiting Caveats

**Important**: While this callback performs optimistic checks before and after LLM execution, rate limits may still be hit in edge cases:

### Why 429 Errors Can Still Occur

The rate limiter uses the following logic:

```python
# TPM satisfied check (optimistic)
tpm_satisfied = self.available_tokens > 0

# Estimated token generation
estimated_gen = self._get_estimated_tokens(messages)

# Actual token count from messages
prompt_tokens = self.get_token_count(messages) if messages else 0
```

**Potential Issues:**

1. **User-Defined Estimation Functions**: `_get_estimated_tokens()` accepts a user-defined method or constant, which cannot guarantee accuracy. This can lead to under- or over-estimation of token usage. While the **Token Adjustment** mechanism corrects estimates based on actual LLM usage reports, there exists a time window where incorrect estimates remain active and may affect other requests when sharing token budgets across concurrent operations.

2. **Timing Window**: There's a brief window between the optimistic check and when the actual LLM provider sees the request. During this window, if another process or concurrent request affects the rate limit state, the check can become invalid.

3. **Token Count Mismatch**: The `get_token_count()` function provided by the caller may use a different tokenizer than the LLM provider itself. This can lead to discrepancies between estimated and actual token usage.

Due to these caveats, users should implement proper retry logic for 429 errors.

### When to Expect Rate Limit Hits

- **Token Estimation Drift**: Models with different encoding schemes than expected
- **Concurrent External Loads**: Other components affecting provider rate limits simultaneously

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
