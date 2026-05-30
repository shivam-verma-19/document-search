import types
from unittest.mock import MagicMock

import pytest

# Mock google.genai before any import so embeddings module loads cleanly
_mock_genai = MagicMock()
_mock_genai_client_instance = MagicMock()
_mock_genai.Client.return_value = _mock_genai_client_instance
import sys

sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", _mock_genai)


class TestRerankerScore:
    def _make_doc(self, text):
        d = types.SimpleNamespace()
        d.page_content = text
        return d

    def test_rerank_returns_sorted_best_first(self):
        from backend.app.reranker import rerank

        docs = [
            self._make_doc("unrelated content about bananas"),
            self._make_doc("python machine learning tutorial python python"),
            self._make_doc("some python code"),
        ]
        result = rerank("python", docs)
        # doc with most "python" occurrences should rank highest
        assert (
            result[0].page_content == "python machine learning tutorial python python"
        )

    def test_rerank_empty_docs(self):
        from backend.app.reranker import rerank

        assert rerank("query", []) == []

    def test_score_length_bonus(self):
        from backend.app.reranker import _score

        short_doc = "python"
        long_doc = " ".join(["python"] * 200)
        s_short = _score("python", short_doc)
        s_long = _score("python", long_doc)
        assert s_long > s_short  # length bonus increases score

    def test_score_no_overlap(self):
        from backend.app.reranker import _score

        score = _score("quantum physics", "banana apple orange")
        assert score == pytest.approx(0.0, abs=0.25)  # length bonus only

    def test_rerank_single_doc(self):
        from backend.app.reranker import rerank

        doc = self._make_doc("hello world")
        assert rerank("hello", [doc]) == [doc]
