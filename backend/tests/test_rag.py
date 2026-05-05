"""
Tests for backend/app/rag.py

All external dependencies (OpenAI, Pinecone, AWS) are stubbed so tests run
fully offline and deterministically.

Covers:
  - rewrite_query   (happy path, LLM error fallback, list response)
  - hybrid_search   (deduplication, k-capping, result type)
  - ask_question    (cache hit, RAG path, fallback path, LLM error)
  - summarize_doc   (delegates to LLM + vector search)
"""

import importlib
import os
import sys
import types
import unittest.mock as mock

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")

# Stub heavy optional deps BEFORE any backend import
from . import _stubs

_stubs.install_all_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(text):
    d = types.SimpleNamespace()
    d.page_content = text
    d.metadata = {}
    return d


def _make_llm(answer="stubbed answer"):
    llm = mock.MagicMock()
    llm.invoke.return_value = mock.MagicMock(content=answer)
    return llm


def _make_vdb(docs=None):
    vdb = mock.MagicMock()
    vdb.similarity_search.return_value = docs or [_doc("context chunk")]
    return vdb


def _make_bm25(docs=None):
    bm25 = mock.MagicMock()
    bm25.search.return_value = docs or []
    return bm25


def _load_rag(
    monkeypatch,
    *,
    cache_hit=None,
    llm_answer="good answer",
    vector_docs=None,
    bm25_docs=None,
    rerank_passthrough=True,
):
    """
    Patch all side effects and return a freshly-imported rag module with
    test doubles injected via the public module globals (llm, vector_db, bm25).
    """
    vector_docs = vector_docs or [_doc(f"doc {i}") for i in range(5)]

    # AWS / metrics stubs
    monkeypatch.setattr("backend.app.metrics.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("backend.app.monitoring.push_metric", lambda *a, **k: None)
    monkeypatch.setattr("backend.app.evaluation.store_eval", lambda *a, **k: None)
    monkeypatch.setattr("backend.app.utils.log_event", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.app.utils.get_secrets", lambda: {"OPENAI_API_KEY": "sk-test"}
    )
    if cache_hit is not None:
        monkeypatch.setattr("backend.app.cache.get_cache", lambda q: cache_hit)
    else:
        monkeypatch.setattr("backend.app.cache.get_cache", lambda q: None)

    monkeypatch.setattr("backend.app.cache.set_cache", lambda q, a: None)

    # reranker
    if rerank_passthrough:
        monkeypatch.setattr("backend.app.reranker.rerank", lambda q, docs: docs)
    else:
        monkeypatch.setattr("backend.app.reranker.rerank", lambda q, docs: [])

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    # Inject test doubles via module globals (picked up by _clients())
    rag_mod._llm = _make_llm(llm_answer)
    rag_mod._vector_db = _make_vdb(vector_docs)
    rag_mod._bm25 = _make_bm25(bm25_docs)

    return rag_mod


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------


class TestRewriteQuery:
    def test_returns_rewritten_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="better query")
        result = rag.rewrite_query("what is AI?")
        assert isinstance(result, str)
        assert result == "better query"

    def test_falls_back_on_llm_exception(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        rag._llm.invoke.side_effect = Exception("network error")  # type: ignore
        result = rag.rewrite_query("original query")
        assert result == "original query"

    def test_handles_list_response(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        rag._llm.invoke.return_value = mock.MagicMock(content=["rewritten", " query"])  # type: ignore
        result = rag.rewrite_query("q")
        assert isinstance(result, str)
        assert "rewritten" in result

    def test_empty_query_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="")
        result = rag.rewrite_query("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_returns_up_to_k_docs(self, monkeypatch):
        docs = [_doc(f"unique text {i}") for i in range(10)]
        rag = _load_rag(monkeypatch, vector_docs=docs[:5], bm25_docs=docs[5:])
        results = rag.hybrid_search("query", k=4)
        assert len(results) == 4

    def test_deduplicates_overlapping_docs(self, monkeypatch):
        shared = _doc("shared content")
        extra_semantic = _doc("only semantic")
        extra_keyword = _doc("only keyword")
        rag = _load_rag(
            monkeypatch,
            vector_docs=[shared, extra_semantic],
            bm25_docs=[shared, extra_keyword],
        )
        results = rag.hybrid_search("q", k=10)
        assert [r.page_content for r in results].count("shared content") == 1

    def test_returns_list(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert isinstance(rag.hybrid_search("q", k=3), list)

    def test_result_count_capped_at_k(self, monkeypatch):
        docs = [_doc(f"doc {i}") for i in range(20)]
        rag = _load_rag(monkeypatch, vector_docs=docs[:10], bm25_docs=docs[10:])
        assert len(rag.hybrid_search("q", k=3)) == 3


# ---------------------------------------------------------------------------
# ask_question
# ---------------------------------------------------------------------------


class TestAskQuestion:
    def test_cache_hit_returns_cached_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, cache_hit="cached!")
        assert rag.ask_question("any query") == "cached!"

    def test_rag_path_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="rag answer")
        assert isinstance(rag.ask_question("what is X?"), str)

    def test_rag_path_returns_llm_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="specific answer")
        assert rag.ask_question("what is X?") == "specific answer"

    def test_fallback_when_rerank_returns_empty(self, monkeypatch):
        """When rerank strips all docs, the LLM fallback path runs."""
        rag = _load_rag(
            monkeypatch,
            vector_docs=[_doc("only one")],
            bm25_docs=[],
            rerank_passthrough=False,
            llm_answer="fallback answer",
        )
        result = rag.ask_question("obscure topic")
        assert result is not None
        assert isinstance(result, str)

    def test_fallback_message_contains_llm_answer(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            vector_docs=[_doc("x")],
            bm25_docs=[],
            rerank_passthrough=False,
            llm_answer="direct llm response",
        )
        result = rag.ask_question("q")
        assert "direct llm response" in result  # type: ignore

    def test_llm_failure_returns_error_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        rag._llm.invoke.side_effect = Exception("LLM down")  # type: ignore
        result = rag.ask_question("anything")
        assert result is not None
        assert isinstance(result, str)

    def test_cache_populated_after_rag(self, monkeypatch):
        written = {}
        # Patch get_cache first so _load_rag picks it up when it reloads the module
        monkeypatch.setattr("backend.app.cache.get_cache", lambda q: None)
        rag = _load_rag(monkeypatch, llm_answer="stored answer")
        # After reload, patch set_cache on the rag module's own imported reference
        rag.set_cache = lambda q, a: written.update({q: a})
        rag.ask_question("will this be cached?")
        assert len(written) >= 1

    def test_does_not_raise_on_empty_query(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="empty ok")
        result = rag.ask_question("")
        assert result is not None


# ---------------------------------------------------------------------------
# summarize_doc
# ---------------------------------------------------------------------------


class TestSummarizeDoc:
    def test_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="the summary")
        assert isinstance(rag.summarize_doc("doc-id-123"), str)

    def test_calls_vector_search_with_doc_id(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="summary")
        rag.summarize_doc("my-doc-id")
        rag._vector_db.similarity_search.assert_called_with("my-doc-id", k=10)  # type: ignore

    def test_returns_llm_content(self, monkeypatch):
        rag = _load_rag(monkeypatch, llm_answer="concise summary text")
        assert rag.summarize_doc("doc") == "concise summary text"
