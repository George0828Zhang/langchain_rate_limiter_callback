import unittest
import logging
from typing import List
from langchain_core.messages import HumanMessage
from langchain_rate_limiter_callback import RateLimiterCallback

# Mock token count
def mock_get_token_count(messages: List[HumanMessage]) -> int:
    return len([m.content for m in messages])

class TestRateLimiterCallback(unittest.TestCase):
    def test_initialization(self):
        callback = RateLimiterCallback(
            get_token_count=mock_get_token_count,
            requests_per_minute=60,
            tokens_per_minute=5000,
            verbose=True
        )
        self.assertEqual(callback.available_requests, 60)
        self.assertEqual(callback.available_tokens, 5000)
        self.assertEqual(callback.rpm_rate, 1.0)
        self.assertEqual(callback.tpm_rate, 5000.0 / 60.0)

    def test_consume_success(self):
        callback = RateLimiterCallback(
            get_token_count=mock_get_token_count,
            requests_per_minute=60,
            tokens_per_minute=5000,
            estimate_generation_token_count=100,
        )
        messages = [HumanMessage(content="Hello")]
        consumed, reason, required = callback._consume(messages)
        self.assertTrue(consumed)
        self.assertIsNone(reason)
        self.assertEqual(required, 101) # 1 (content) + 100 (estimate)
        self.assertEqual(callback.available_requests, 59)
        self.assertEqual(callback.available_tokens, 5000 - 101)

    def test_consume_rpm_limit(self):
        callback = RateLimiterCallback(
            get_token_count=mock_get_token_count,
            requests_per_minute=1,
            tokens_per_minute=5000,
            estimate_generation_token_count=100,
        )
        # First consumption
        messages = [HumanMessage(content="Hello")]
        consumed, reason, required = callback._consume(messages)
        self.assertTrue(consumed)
        
        # Second consumption should fail
        consumed, reason, required = callback._consume(messages)
        self.assertFalse(consumed)
        self.assertEqual(reason, "RPM limit reached")

if __name__ == "__main__":
    unittest.main()
