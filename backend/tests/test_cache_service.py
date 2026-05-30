import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", MagicMock())

import backend.app.cache_service as m  # noqa: E402


class TestCacheService:
    def test_get_cached_answer_hit(self):
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = "cached answer"
        import backend.app

        with patch.object(backend.app, "cache", mock_cache):
            result = m.get_cached_answer("my query")
        assert result == "cached answer"

    def test_get_cached_answer_miss(self):
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = None
        with patch.dict(sys.modules, {"backend.app.cache": mock_cache}):
            result = m.get_cached_answer("my query")
        assert result is None

    def test_get_cached_answer_exception_returns_none(self):
        mock_cache = MagicMock()
        mock_cache.get_cache.side_effect = Exception("DynamoDB down")
        with patch.dict(sys.modules, {"backend.app.cache": mock_cache}):
            result = m.get_cached_answer("query")
        assert result is None

    def test_set_cached_answer_success(self):
        mock_cache = MagicMock()
        mock_cache.set_cache.return_value = None
        import backend.app

        with patch.object(backend.app, "cache", mock_cache):
            result = m.set_cached_answer("query", "answer")
        assert result is True

    def test_set_cached_answer_exception_returns_false(self):
        mock_cache = MagicMock()
        mock_cache.set_cache.side_effect = Exception("write error")
        with patch.dict(sys.modules, {"backend.app.cache": mock_cache}):
            result = m.set_cached_answer("query", "answer")
        assert result is False
