"""
Tests for backend/app/rag.py

Updated for:
- OpenSearch Serverless
- Embedding-based retrieval
- Hybrid search
- Reranking
- Cache handling
- LLM fallback handling

All external services are mocked.
"""

import importlib
import os
import types
import unittest.mock as mock
from typing import cast

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")

from . import _stubs

_stubs.install_all_stubs()


# =========================================================
# HELPERS
# =========================================================


def _doc(text):
    d = types.SimpleNamespace()
    d.page_content = text
    d.metadata = {}
    return d


def _make_llm(answer="stubbed answer"):
    llm = mock.MagicMock()
    llm.invoke.return_value = mock.MagicMock(content=answer)
    return llm


def _load_rag(
    monkeypatch,
    *,
    cache_hit=None,
    llm_answer="good answer",
    search_results=None,
    rerank_passthrough=True,
):
    """
    Reload rag.py with fully mocked dependencies.
    """

    if search_results is None:
        search_results = [
            "retrieved chunk 1",
            "retrieved chunk 2",
            "retrieved chunk 3",
        ]

    # =====================================================
    # Mock metrics / monitoring
    # =====================================================

    monkeypatch.setattr(
        "backend.app.metrics.log_metrics",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "backend.app.monitoring.push_metric",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "backend.app.evaluation.store_eval",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "backend.app.utils.log_event",
        lambda *a, **k: None,
    )

    # =====================================================
    # Cache
    # =====================================================

    if cache_hit is not None:
        monkeypatch.setattr(
            "backend.app.cache.get_cache",
            lambda q: cache_hit,
        )
    else:
        monkeypatch.setattr(
            "backend.app.cache.get_cache",
            lambda q: None,
        )

    monkeypatch.setattr(
        "backend.app.cache.set_cache",
        lambda q, a: None,
    )

    # =====================================================
    # Embeddings
    # =====================================================

    monkeypatch.setattr(
        "backend.app.embeddings.get_embedding",
        lambda q: [0.1, 0.2, 0.3],
    )

    # =====================================================
    # OpenSearch
    # =====================================================

    monkeypatch.setattr(
        "backend.app.opensearch_client.search_similar",
        lambda embedding, k=5: search_results[:k],
    )

    # =====================================================
    # Reranker
    # =====================================================

    if rerank_passthrough:
        monkeypatch.setattr(
            "backend.app.reranker.rerank",
            lambda q, docs: docs,
        )
    else:
        monkeypatch.setattr(
            "backend.app.reranker.rerank",
            lambda q, docs: [],
        )

    # =====================================================
    # Reload rag
    # =====================================================

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    # Inject mocked LLM
    rag_mod._llm = _make_llm(llm_answer)

    return rag_mod


# =========================================================
# rewrite_query
# =========================================================


class TestRewriteQuery:
    def test_returns_rewritten_string(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            llm_answer="better rewritten query",
        )

        result = rag.rewrite_query("what is AI?")

        assert isinstance(result, str)
        assert result == "better rewritten query"

    def test_falls_back_on_exception(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        cast(mock.MagicMock, rag._llm.invoke).side_effect = Exception("network error")

        result = rag.rewrite_query("original query")

        assert result == "original query"

    def test_empty_query_returns_empty_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.rewrite_query("")

        assert result == ""


# =========================================================
# hybrid_search
# =========================================================


class TestHybridSearch:
    def test_returns_docs(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        docs = rag.hybrid_search("query", k=3)

        assert isinstance(docs, list)
        assert len(docs) == 3

    def test_docs_have_page_content(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        docs = rag.hybrid_search("query")

        assert hasattr(docs[0], "page_content")

    def test_caps_results_at_k(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=[
                "a",
                "b",
                "c",
                "d",
                "e",
            ],
        )

        docs = rag.hybrid_search("query", k=2)

        assert len(docs) == 2

    def test_empty_query_returns_empty_list(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        docs = rag.hybrid_search("", k=5)

        assert docs == []


# =========================================================
# ask_question
# =========================================================


class TestAskQuestion:
    def test_cache_hit_returns_cached_answer(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            cache_hit="cached response",
        )

        result = rag.ask_question("query")

        assert result == "cached response"

    def test_rag_path_returns_answer(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            llm_answer="rag answer",
        )

        result = rag.ask_question("what is AI?")

        assert result == "rag answer"

    def test_fallback_path_runs(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["single result"],
            rerank_passthrough=False,
            llm_answer="fallback answer",
        )

        result = rag.ask_question("obscure question")

        assert "fallback answer" in str(result)

    def test_forbidden_query_blocked(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.ask_question("how to hack database")

        assert result == "Query not allowed"

    def test_empty_query_returns_empty_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.ask_question("")

        assert result == ""

    def test_llm_failure_returns_error(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        cast(mock.MagicMock, rag._llm.invoke).side_effect = Exception("LLM down")

        result = rag.ask_question("question")

        assert isinstance(result, str)

    def test_cache_written_after_rag(self, monkeypatch):
        written = {}

        monkeypatch.setattr(
            "backend.app.cache.get_cache",
            lambda q: None,
        )

        rag = _load_rag(
            monkeypatch,
            llm_answer="stored answer",
        )

        rag.set_cache = lambda q, a: written.update({q: a})

        rag.ask_question("cache this")

        assert len(written) > 0

    def test_forbidden_query_is_case_insensitive(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.ask_question("HACK the system")

        assert result == "Query not allowed"

    def test_retry_exhausted_returns_error_string(self, monkeypatch):
        """LLM always raises and docs < 2 → error string, not an exception."""
        rag = _load_rag(
            monkeypatch,
            search_results=[],  # triggers fallback path (docs < 2)
            llm_answer="irrelevant",
        )
        cast(mock.MagicMock, rag._llm.invoke).side_effect = Exception("LLM down")

        result = rag.ask_question("what is X?")

        assert "Error" in str(result)

    def test_fallback_path_writes_cache(self, monkeypatch):
        # 1. Load the rag module
        rag = _load_rag(
            monkeypatch,
            search_results=[],  # forces docs < 2 → fallback
            llm_answer="fallback answer",
        )

        written = {}
        # 2. Patch the reference INSIDE the reloaded rag module specifically
        monkeypatch.setattr(rag, "set_cache", lambda q, a: written.update({q: a}))

        rag.ask_question("any question")
        assert len(written) == 1

    def test_exactly_one_doc_triggers_fallback_not_rag(self, monkeypatch):
        """Boundary: 1 doc < 2 threshold → fallback prefix appears in answer."""
        rag = _load_rag(
            monkeypatch,
            search_results=["single chunk"],
            rerank_passthrough=True,
            llm_answer="llm says hi",
        )

        result = rag.ask_question("question?")

        assert "Switching to LLM" in str(result)

    def test_two_docs_takes_rag_path_not_fallback(self, monkeypatch):
        """Boundary: 2 docs >= 2 threshold → RAG path, no fallback prefix."""
        rag = _load_rag(
            monkeypatch,
            search_results=["chunk a", "chunk b"],
            rerank_passthrough=True,
            llm_answer="rag answer",
        )

        result = rag.ask_question("question?")

        assert "Switching to LLM" not in str(result)


# =========================================================
# summarize_doc
# =========================================================


class TestSummarizeDoc:
    def test_returns_summary(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            llm_answer="summary text",
        )

        result = rag.summarize_doc("doc-id")

        assert result == "summary text"

    def test_returns_no_content_when_empty(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=[],
        )

        result = rag.summarize_doc("missing-doc")

        assert result == "No content found."

    def test_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.summarize_doc("doc")

        assert isinstance(result, str)

    def test_handles_llm_failure(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        cast(mock.MagicMock, rag._llm.invoke).side_effect = Exception("LLM down")

        result = rag.summarize_doc("doc")

        assert result == "Error generating summary."
