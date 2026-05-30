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


class TestErrorSeverityAndCategory:
    def test_error_severity_values(self):
        from backend.app.errors import ErrorSeverity

        assert ErrorSeverity.CRITICAL == "critical"
        assert ErrorSeverity.ERROR == "error"
        assert ErrorSeverity.WARNING == "warning"
        assert ErrorSeverity.INFO == "info"

    def test_error_category_values(self):
        from backend.app.errors import ErrorCategory

        assert ErrorCategory.AUTH == "authentication"
        assert ErrorCategory.VALIDATION == "validation"
        assert ErrorCategory.RATE_LIMIT == "rate_limit"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.RESOURCE_NOT_FOUND == "not_found"
        assert ErrorCategory.CACHE == "cache"
        assert ErrorCategory.UNKNOWN == "unknown"


class TestRAGException:
    def test_default_fields(self):
        from backend.app.errors import ErrorCategory, ErrorSeverity, RAGException

        exc = RAGException("something went wrong")
        assert exc.message == "something went wrong"
        assert exc.category == ErrorCategory.UNKNOWN
        assert exc.severity == ErrorSeverity.ERROR
        assert exc.status_code == 500
        assert exc.retryable is False
        assert exc.details == {}
        assert exc.original_exception is None
        assert isinstance(exc.timestamp, str)

    def test_to_dict_shape(self):
        from backend.app.errors import RAGException

        exc = RAGException("test", details={"key": "val"})
        d = exc.to_dict()
        assert "error" in d
        inner = d["error"]
        assert inner["message"] == "test"
        assert inner["retryable"] is False
        assert inner["details"] == {"key": "val"}

    def test_log_critical(self, caplog):
        import logging

        from backend.app.errors import ErrorCategory, ErrorSeverity, RAGException

        exc = RAGException(
            "boom", severity=ErrorSeverity.CRITICAL, category=ErrorCategory.AUTH
        )
        with caplog.at_level(logging.CRITICAL, logger="backend.app.errors"):
            exc.log()
        assert any(
            "AUTHENTICATION" in r.message or "boom" in r.message for r in caplog.records
        )

    def test_log_warning(self, caplog):
        import logging

        from backend.app.errors import ErrorSeverity, RAGException

        exc = RAGException("warn", severity=ErrorSeverity.WARNING)
        with caplog.at_level(logging.WARNING, logger="backend.app.errors"):
            exc.log()

    def test_log_info(self, caplog):
        import logging

        from backend.app.errors import ErrorSeverity, RAGException

        exc = RAGException("info msg", severity=ErrorSeverity.INFO)
        with caplog.at_level(logging.INFO, logger="backend.app.errors"):
            exc.log()


class TestSubclassedExceptions:
    def test_authentication_error(self):
        from backend.app.errors import AuthenticationError

        exc = AuthenticationError()
        assert exc.status_code == 401
        assert exc.retryable is False

    def test_validation_error(self):
        from backend.app.errors import ValidationError

        exc = ValidationError("bad input")
        assert exc.status_code == 400
        assert exc.retryable is False

    def test_rate_limit_error_sets_retry_after(self):
        from backend.app.errors import RateLimitError

        exc = RateLimitError(retry_after=30)
        assert exc.status_code == 429
        assert exc.retryable is True
        assert exc.details["retry_after"] == 30

    def test_timeout_error_populates_details(self):
        from backend.app.errors import TimeoutError

        exc = TimeoutError(service="embeddings", timeout_ms=5000)
        assert exc.status_code == 504
        assert exc.retryable is True
        assert exc.details["service"] == "embeddings"
        assert exc.details["timeout_ms"] == 5000

    def test_resource_not_found(self):
        from backend.app.errors import ResourceNotFoundError

        exc = ResourceNotFoundError(resource="Document", resource_id="abc")
        assert exc.status_code == 404
        assert "Document" in exc.message
        assert exc.details["resource"] == "Document"

    def test_cache_error(self):
        from backend.app.errors import CacheError

        exc = CacheError(operation="get")
        assert exc.status_code == 500
        assert exc.retryable is True
        assert exc.details["operation"] == "get"


class TestErrorResponse:
    def test_from_exception(self):
        from backend.app.errors import AuthenticationError, ErrorResponse

        exc = AuthenticationError("test auth fail")
        resp = ErrorResponse.from_exception(exc)
        assert resp.status_code == 401
        assert resp.error["message"] == "test auth fail"

    def test_from_unknown_error(self):
        from backend.app.errors import ErrorResponse

        resp = ErrorResponse.from_unknown_error(ValueError("oops"))
        assert resp.status_code == 500
        assert resp.error["category"] == "unknown"
        assert resp.error["details"]["error_type"] == "ValueError"


class TestRetryConfig:
    def test_get_delay_ms_exponential(self):
        from backend.app.errors import RetryConfig

        cfg = RetryConfig(
            initial_delay_ms=100, backoff_multiplier=2.0, max_delay_ms=5000
        )
        assert cfg.get_delay_ms(1) == 100
        assert cfg.get_delay_ms(2) == 200
        assert cfg.get_delay_ms(3) == 400

    def test_get_delay_ms_capped(self):
        from backend.app.errors import RetryConfig

        cfg = RetryConfig(
            initial_delay_ms=1000, backoff_multiplier=10.0, max_delay_ms=3000
        )
        assert cfg.get_delay_ms(5) == 3000


class TestErrorTracker:
    def test_record_and_get_stats(self):
        from backend.app.errors import ErrorCategory, ErrorSeverity, ErrorTracker

        tracker = ErrorTracker()
        tracker.record(ErrorCategory.AUTH, ErrorSeverity.WARNING)
        tracker.record(ErrorCategory.AUTH, ErrorSeverity.WARNING)
        stats = tracker.get_stats()
        assert stats["total_errors"] == 2
        assert "authentication:warning" in stats["by_category"]

    def test_reset_clears_data(self):
        from backend.app.errors import ErrorCategory, ErrorSeverity, ErrorTracker

        tracker = ErrorTracker()
        tracker.record(ErrorCategory.CACHE, ErrorSeverity.ERROR)
        tracker.reset()
        assert tracker.get_stats()["total_errors"] == 0

    def test_critical_severity_logs(self, caplog):
        import logging

        from backend.app.errors import ErrorCategory, ErrorSeverity, ErrorTracker

        tracker = ErrorTracker()
        with caplog.at_level(logging.CRITICAL, logger="backend.app.errors"):
            tracker.record(ErrorCategory.AUTH, ErrorSeverity.CRITICAL)
        assert any("Critical" in r.message for r in caplog.records)
