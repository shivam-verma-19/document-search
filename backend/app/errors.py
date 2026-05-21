# backend/app/errors.py
"""
Custom exception classes and error handling utilities for the RAG platform.

This module provides:
- Custom exception types for different failure scenarios
- Structured error responses
- Error tracking and monitoring
- Retry strategies
"""

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR TYPES AND SEVERITIES
# ─────────────────────────────────────────────────────────────────────────────


class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""

    CRITICAL = "critical"  # System down, immediate action required
    ERROR = "error"  # Serious error, feature broken
    WARNING = "warning"  # Degraded functionality, may self-recover
    INFO = "info"  # Informational, no action needed


class ErrorCategory(str, Enum):
    """Error categories for routing and handling."""

    AUTH = "authentication"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    RESOURCE_NOT_FOUND = "not_found"
    BEDROCK = "bedrock"
    FAISS = "faiss"
    CACHE = "cache"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────


class RAGException(Exception):
    """Base exception for RAG platform."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        status_code: int = 500,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow().isoformat()

        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to API response format."""
        return {
            "error": {
                "message": self.message,
                "category": self.category.value,
                "severity": self.severity.value,
                "retryable": self.retryable,
                "timestamp": self.timestamp,
                "details": self.details,
            }
        }

    def log(self):
        """Log the exception with appropriate level."""
        log_message = (
            f"[{self.category.value.upper()}] {self.message} | "
            f"Status: {self.status_code} | Retryable: {self.retryable}"
        )

        if self.details:
            log_message += f" | Details: {self.details}"

        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=self.original_exception)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message, exc_info=self.original_exception)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message, exc_info=self.original_exception)
        else:
            logger.info(log_message)


class AuthenticationError(RAGException):
    """User authentication failed."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTH,
            severity=ErrorSeverity.WARNING,
            status_code=401,
            retryable=False,
            **kwargs,
        )


class ValidationError(RAGException):
    """Input validation failed."""

    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            status_code=400,
            retryable=False,
            **kwargs,
        )


class RateLimitError(RAGException):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            status_code=429,
            retryable=True,
            details=details,
            **kwargs,
        )


class TimeoutError(RAGException):
    """Request timed out."""

    def __init__(
        self,
        message: str = "Request timed out",
        service: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if service:
            details["service"] = service
        if timeout_ms:
            details["timeout_ms"] = timeout_ms

        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            status_code=504,
            retryable=True,
            details=details,
            **kwargs,
        )


class ResourceNotFoundError(RAGException):
    """Resource not found."""

    def __init__(self, resource: str, resource_id: str, **kwargs):
        super().__init__(
            message=f"{resource} not found: {resource_id}",
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.WARNING,
            status_code=404,
            retryable=False,
            details={"resource": resource, "resource_id": resource_id},
            **kwargs,
        )


class BedrockError(RAGException):
    """Bedrock API error."""

    def __init__(
        self,
        message: str = "Bedrock API error",
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if model:
            details["model"] = model
        if error_code:
            details["error_code"] = error_code

        retryable = error_code in [
            "ThrottlingException",
            "RequestLimitExceeded",
            "ServiceUnavailableException",
            "InternalServerError",
        ]

        super().__init__(
            message=message,
            category=ErrorCategory.BEDROCK,
            severity=ErrorSeverity.ERROR,
            status_code=503 if retryable else 500,
            retryable=retryable,
            details=details,
            **kwargs,
        )


class FAISSError(RAGException):
    """FAISS/vector store error."""

    def __init__(self, message: str = "Vector store error", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.FAISS,
            severity=ErrorSeverity.ERROR,
            status_code=503,
            retryable=True,
            **kwargs,
        )


class CacheError(RAGException):
    """Cache operation error."""

    def __init__(
        self, message: str = "Cache error", operation: Optional[str] = None, **kwargs
    ):
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.WARNING,
            status_code=500,
            retryable=True,
            details=details,
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ERROR RESPONSE FORMATTER
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ErrorResponse:
    """Structured error response sent to client."""

    error: Dict[str, Any]
    status_code: int

    @classmethod
    def from_exception(cls, exc: RAGException) -> "ErrorResponse":
        """Create error response from exception."""
        exc.log()
        return cls(
            error=exc.to_dict()["error"],
            status_code=exc.status_code,
        )

    @classmethod
    def from_unknown_error(
        cls,
        error: Exception,
        status_code: int = 500,
        user_message: str = "An unexpected error occurred",
    ) -> "ErrorResponse":
        """Create error response from unknown exception."""
        logger.error(
            f"Unexpected error: {str(error)}",
            exc_info=error,
        )

        return cls(
            error={
                "message": user_message,
                "category": "unknown",
                "severity": "error",
                "retryable": False,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": type(error).__name__,
                },
            },
            status_code=status_code,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RETRY STRATEGY
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple = (
        TimeoutError,
        RateLimitError,
        BedrockError,
        FAISSError,
    )

    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay_ms * (self.backoff_multiplier ** (attempt - 1))
        return min(int(delay), self.max_delay_ms)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR TRACKING AND MONITORING
# ─────────────────────────────────────────────────────────────────────────────


class ErrorTracker:
    """Track errors for monitoring and alerting."""

    def __init__(self):
        self.errors: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}

    def record(self, category: ErrorCategory, severity: ErrorSeverity):
        """Record an error occurrence."""
        key = f"{category.value}:{severity.value}"
        self.errors[key] = self.errors.get(key, 0) + 1
        self.last_error_time[key] = datetime.utcnow().timestamp()

        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error recorded: {key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.errors.values()),
            "by_category": self.errors,
            "last_errors": self.last_error_time,
        }

    def reset(self):
        """Reset error tracking."""
        self.errors.clear()
        self.last_error_time.clear()


error_tracker = ErrorTracker()
