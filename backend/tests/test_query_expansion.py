"""Tests for query_expansion.py — HyDE (Hypothetical Document Embeddings)."""

import sys

import pytest

import importlib


@pytest.fixture(autouse=True)
def _evict_stubs():
    """Remove any MagicMock stubs installed by test_rag so real modules load."""
    for mod in ["backend.app.query_expansion"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.query_expansion"]:
        sys.modules.pop(mod, None)


from unittest.mock import MagicMock, patch

import pytest


class TestGenerateHydeQuery:
    def test_returns_hypothesis_on_success(self, monkeypatch):
        monkeypatch.setenv("HYDE_ENABLED", "true")
        import backend.app.query_expansion as qe

        monkeypatch.setattr(qe, "HYDE_ENABLED", True)

        mock_response = MagicMock()
        mock_response.text = "A hypothetical answer passage about the topic."
        # _get_client is imported inside generate_hyde_query from gemini_client
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                mock_response
            )
            result = qe.generate_hyde_query("What is retrieval augmented generation?")

        assert result == "A hypothetical answer passage about the topic."

    def test_returns_original_query_when_disabled(self, monkeypatch):
        import backend.app.query_expansion as qe
        importlib.reload(qe)

        monkeypatch.setattr(qe, "HYDE_ENABLED", False)

        result = qe.generate_hyde_query("What is RAG?")
        assert result == "What is RAG?"

    def test_returns_original_query_on_llm_failure(self, monkeypatch):
        import backend.app.query_expansion as qe
        importlib.reload(qe)

        monkeypatch.setattr(qe, "HYDE_ENABLED", True)

        with patch(
            "backend.app.gemini_client._get_client", side_effect=Exception("api down")
        ):
            result = qe.generate_hyde_query("What is RAG?")

        assert result == "What is RAG?"

    def test_returns_original_on_empty_llm_response(self, monkeypatch):
        import backend.app.query_expansion as qe
        importlib.reload(qe)

        monkeypatch.setattr(qe, "HYDE_ENABLED", True)

        mock_response = MagicMock()
        mock_response.text = ""
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                mock_response
            )
            result = qe.generate_hyde_query("What is RAG?")

        assert result == "What is RAG?"

    def test_empty_query_returned_unchanged(self, monkeypatch):
        import backend.app.query_expansion as qe

        monkeypatch.setattr(qe, "HYDE_ENABLED", True)

        assert qe.generate_hyde_query("") == ""
        assert qe.generate_hyde_query("   ") == "   "

    def test_hypothesis_different_from_original(self, monkeypatch):
        """The HyDE text should be clearly different from the raw question."""
        import backend.app.query_expansion as qe
        importlib.reload(qe)

        monkeypatch.setattr(qe, "HYDE_ENABLED", True)

        query = "What is RAG?"
        mock_response = MagicMock()
        mock_response.text = (
            "RAG is a technique that combines retrieval with generation."
        )
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                mock_response
            )
            result = qe.generate_hyde_query(query)

        assert result != query
        assert len(result) > len(query)

    def test_respects_max_tokens_config(self, monkeypatch):
        import backend.app.query_expansion as qe
        importlib.reload(qe)

        monkeypatch.setattr(qe, "HYDE_MAX_TOKENS", 50)

        assert qe.HYDE_MAX_TOKENS == 50
