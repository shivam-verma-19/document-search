import os
from unittest.mock import MagicMock, patch

import pytest

# Mock google.genai before any import so embeddings module loads cleanly
_mock_genai = MagicMock()
_mock_genai_client_instance = MagicMock()
_mock_genai.Client.return_value = _mock_genai_client_instance
import sys

sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", _mock_genai)


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        from backend.app.retry import retry_with_backoff

        result = retry_with_backoff(lambda: "ok")
        assert result == "ok"

    def test_retries_on_transient_error_then_succeeds(self):
        from backend.app.retry import retry_with_backoff

        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("timeout error")
            return "done"

        with patch("time.sleep"):
            result = retry_with_backoff(fn, max_retries=3, base_delay_ms=10)
        assert result == "done"
        assert call_count[0] == 3

    def test_raises_after_max_retries_exceeded(self):
        from backend.app.retry import retry_with_backoff

        def always_fail():
            raise Exception("503 error")

        with patch("time.sleep"):
            with pytest.raises(Exception, match="503 error"):
                retry_with_backoff(always_fail, max_retries=2, base_delay_ms=10)

    def test_non_retryable_error_raises_immediately(self):
        from backend.app.retry import retry_with_backoff

        call_count = [0]

        def fn():
            call_count[0] += 1
            raise ValueError("validation error")

        with pytest.raises(ValueError):
            retry_with_backoff(fn, max_retries=3, base_delay_ms=10)

        assert call_count[0] == 1  # no retries for non-retryable
