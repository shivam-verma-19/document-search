import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock google.genai before importing the module under test
_mock_genai = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = _mock_genai

import backend.app.embeddings as m  # noqa: E402


def _reset():
    m._client = None


class TestGetEmbedding:
    def _make_response(self, values):
        embedding = MagicMock()
        embedding.values = values
        response = MagicMock()
        response.embeddings = [embedding]
        return response

    def test_returns_float_list(self):
        _reset()
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_response(
            [0.1, 0.2, 0.3]
        )
        with patch.object(m, "_client", mock_client):
            result = m.get_embedding("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_empty_text_returns_empty_list(self):
        _reset()
        result = m.get_embedding("")
        assert result == []

    def test_raises_on_api_error(self):
        _reset()
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("API error")
        with patch.object(m, "_client", mock_client):
            with pytest.raises(RuntimeError, match="Embedding failed"):
                m.get_embedding("some text")

    def test_no_api_key_raises(self):
        _reset()
        with patch("backend.app.secrets.get_secret", return_value=""):
            with pytest.raises((ValueError, RuntimeError)):
                m._get_client()
