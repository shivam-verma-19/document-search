"""
Tests for query_planner.py — multi-step query decomposition.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _evict_stubs():
    for mod in ["backend.app.query_planner"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.query_planner"]:
        sys.modules.pop(mod, None)


class TestQueryDecomposition:
    def test_empty_query_returns_empty(self):
        from backend.app.query_planner import decompose_query

        assert decompose_query("") == []
        assert decompose_query("   ") == []

    def test_simple_query_bypasses_decomposition(self):
        from backend.app.gemini_client import ClassifierResult
        from backend.app.query_planner import decompose_query

        simple_res = ClassifierResult(complexity="simple", score=0.1, signals=["what is"])
        with patch("backend.app.query_planner.classify_complexity", return_value=simple_res):
            result = decompose_query("What is S3 Vectors?")
            assert result == ["What is S3 Vectors?"]

    def test_complex_query_decomposes_into_subqueries(self):
        from backend.app.gemini_client import ClassifierResult
        from backend.app.query_planner import decompose_query

        complex_res = ClassifierResult(complexity="complex", score=0.9, signals=["compare"])
        mock_resp = MagicMock()
        mock_resp.text = "1. AWS Lambda scaling and concurrency limits\n2. DynamoDB auto-scaling and pricing"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp

        with patch("backend.app.query_planner.classify_complexity", return_value=complex_res), \
             patch("backend.app.query_planner._get_client", return_value=mock_client):
            result = decompose_query("Compare AWS Lambda and DynamoDB scaling")
            assert len(result) == 2
            assert "AWS Lambda" in result[0]
            assert "DynamoDB" in result[1]

    def test_disabled_decomposition_returns_original(self, monkeypatch):
        monkeypatch.setenv("QUERY_DECOMPOSITION_ENABLED", "false")
        import importlib
        import backend.app.query_planner as qp
        importlib.reload(qp)

        assert qp.decompose_query("Compare A and B") == ["Compare A and B"]

    def test_fallback_on_llm_error(self):
        from backend.app.gemini_client import ClassifierResult
        from backend.app.query_planner import decompose_query

        complex_res = ClassifierResult(complexity="complex", score=0.9, signals=["compare"])
        with patch("backend.app.query_planner.classify_complexity", return_value=complex_res), \
             patch("backend.app.query_planner._get_client", side_effect=Exception("API error")):
            result = decompose_query("Compare A and B")
            assert result == ["Compare A and B"]
