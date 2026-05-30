import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_cached_answer(query: str) -> Optional[str]:
    from . import cache

    try:
        result = cache.get_cache(query)
        if result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return str(result)
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None


def set_cached_answer(query: str, answer: str) -> bool:
    from . import cache

    try:
        cache.set_cache(query, answer)
        return True
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
        return False
