import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def retry_with_backoff(
    fn: Callable[[], Any],
    retryable_errors: list[str] | None = None,
    max_retries: int = 3,
    base_delay_ms: int = 100,
):
    retryable_errors = retryable_errors or [
        "timeout",
        "throttl",
        "503",
        "502",
        "500",
        "unavailable",
        "rate limit",
    ]

    attempt = 0
    last_error = None

    while attempt < max_retries:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(err in error_str for err in retryable_errors)

            if not retryable or attempt >= max_retries:
                logger.error(
                    f"Retry failed after {attempt} attempts: {e}", exc_info=True
                )
                raise

            delay = min(base_delay_ms * (2 ** (attempt - 1)), 5000) / 1000
            logger.warning(f"Transient error, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)

    # Ensure we raise a valid BaseException. last_error may be None or a non-exception
    if isinstance(last_error, BaseException):
        raise last_error
    raise RuntimeError("Retry failed without an exception")
