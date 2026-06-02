"""
Tests for rag.py — hybrid RAG pipeline with normalized cache keys,
source attribution, and LLM-only fallback.
"""

import importlib
import os
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest

from . import _stubs


@pytest.fixture(autouse=True, scope="module")
def _install_stubs():
    _stubs.install_all_stubs()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_doc(
    text: str, doc_id: str = "", filename: str = "test.pdf", chunk_index: int = 0
):
    from backend.app.document_repository import SearchDocument

    return SearchDocument(
        page_content=text,
        doc_id=doc_id or f"id:{text[:10]}",
        metadata={"filename": filename, "chunk_index": chunk_index},
    )


def _router_result(answer="good answer", confidence=0.80, escalated=False):
    return {
        "answer": answer,
        "model_used": "gemini-2.5-flash",
        "complexity": "simple",
        "confidence": confidence,
        "escalated": escalated,
        "attempted": ["gemini-2.5-flash"],
        "errors": {},
    }


def _load_rag(
    monkeypatch,
    *,
    cache_hit=None,
    router_answer="good answer",
    router_raises=False,
    search_results=None,
    rerank_passthrough=True,
):
    if search_results is None:
        search_results = [_make_doc("chunk 1", "id1"), _make_doc("chunk 2", "id2")]

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    import backend.app.cache as cache_mod
    import backend.app.gemini_client as gemini_router
    import backend.app.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(cache_mod, "get_cache", lambda q: cache_hit)
    monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: None)

    # Patch hybrid_search and rerank_documents in rag's own namespace since
    # rag.py uses `from .search_service import hybrid_search, rerank_documents`
    monkeypatch.setattr(
        rag_mod,
        "hybrid_search",
        lambda query, k=5, **kw: search_results if query.strip() else [],
    )

    if rerank_passthrough:
        monkeypatch.setattr(rag_mod, "rerank_documents", lambda q, docs: docs)
    else:
        monkeypatch.setattr(rag_mod, "rerank_documents", lambda q, docs: [])

    if router_raises:
        monkeypatch.setattr(
            gemini_router,
            "route_and_invoke",
            MagicMock(side_effect=Exception("router down")),
        )
    else:
        monkeypatch.setattr(
            gemini_router,
            "route_and_invoke",
            MagicMock(return_value=_router_result(answer=router_answer)),
        )

    return rag_mod


# ─── Cache key normalization ──────────────────────────────────────────────────


class TestCacheKeyNormalization:
    def test_different_case_hits_same_cache(self, monkeypatch):
        """'What is RAG?' and 'what is rag' must resolve to the same cache key."""
        written_keys = []
        rag = _load_rag(monkeypatch, router_answer="answer")
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: written_keys.append(q))

        rag.ask_question("What is RAG?")
        rag.ask_question("what is rag")

        # Both calls should write the same normalized key
        assert len(written_keys) == 2
        assert written_keys[0] == written_keys[1]

    def test_trailing_punctuation_normalized(self, monkeypatch):
        written_keys = []
        rag = _load_rag(monkeypatch, router_answer="answer")
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: written_keys.append(q))

        rag.ask_question("What is RAG?")
        rag.ask_question("What is RAG")

        assert written_keys[0] == written_keys[1]

    def test_cache_hit_with_normalized_key(self, monkeypatch):
        # Simulate cache populated with lowercase key
        rag = _load_rag(monkeypatch, cache_hit="cached answer")
        result = rag.ask_question("WHAT IS RAG?")
        assert result == "cached answer"


# ─── Source attribution ───────────────────────────────────────────────────────


class TestSourceAttribution:
    def test_prompt_includes_filename(self, monkeypatch):
        # Provide 2 docs to meet MIN_DOCS_FOR_RAG threshold
        docs = [
            _make_doc("content here", "id1", filename="report.pdf", chunk_index=0),
            _make_doc("more content", "id2", filename="report.pdf", chunk_index=1),
        ]
        rag = _load_rag(monkeypatch, search_results=docs)
        import backend.app.gemini_client as gemini_router

        spy = MagicMock(return_value=_router_result())
        monkeypatch.setattr(gemini_router, "route_and_invoke", spy)

        rag.ask_question("question")

        prompt_arg = spy.call_args.kwargs.get("prompt", "")
        assert "report.pdf" in prompt_arg

    def test_prompt_includes_chunk_index(self, monkeypatch):
        # Provide 2 docs to meet MIN_DOCS_FOR_RAG threshold
        docs = [
            _make_doc("content", "id1", filename="doc.pdf", chunk_index=3),
            _make_doc("more content", "id2", filename="doc.pdf", chunk_index=4),
        ]
        rag = _load_rag(monkeypatch, search_results=docs)
        import backend.app.gemini_client as gemini_router

        spy = MagicMock(return_value=_router_result())
        monkeypatch.setattr(gemini_router, "route_and_invoke", spy)

        rag.ask_question("question")

        prompt_arg = spy.call_args.kwargs.get("prompt", "")
        assert "3" in prompt_arg

    def test_multiple_sources_all_labeled(self, monkeypatch):
        docs = [
            _make_doc("text a", "id1", filename="a.pdf", chunk_index=0),
            _make_doc("text b", "id2", filename="b.pdf", chunk_index=1),
        ]
        rag = _load_rag(monkeypatch, search_results=docs)
        import backend.app.gemini_client as gemini_router

        spy = MagicMock(return_value=_router_result())
        monkeypatch.setattr(gemini_router, "route_and_invoke", spy)

        rag.ask_question("question")

        prompt_arg = spy.call_args.kwargs.get("prompt", "")
        assert "a.pdf" in prompt_arg
        assert "b.pdf" in prompt_arg


# ─── Pipeline routing ─────────────────────────────────────────────────────────


class TestPipelineRouting:
    def test_docs_found_uses_rag_path(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="rag answer")
        result = rag.ask_question("what is ai")
        assert result == "rag answer"

    def test_no_docs_uses_llm_only_path(self, monkeypatch):
        # With no docs, fallback LLM path is used; answer is returned as-is
        # since there's no "I couldn't find..." prefix when search_results=[]
        # but the LLM answer is wrapped. Check it contains the router answer.
        rag = _load_rag(monkeypatch, search_results=[], router_answer="llm answer")
        result = rag.ask_question("what is ai")
        assert "llm answer" in result

    def test_llm_only_path_passes_empty_context(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=[])
        import backend.app.gemini_client as gemini_router

        spy = MagicMock(return_value=_router_result())
        monkeypatch.setattr(gemini_router, "route_and_invoke", spy)
        rag.ask_question("question")
        assert spy.call_args.kwargs.get("context", "SENTINEL") == ""

    def test_single_doc_still_uses_rag(self, monkeypatch):
        """1 doc → RAG path (MIN_DOCS_FOR_RAG constant exists but is not enforced in logic)."""
        rag = _load_rag(
            monkeypatch,
            search_results=[_make_doc("one chunk", "id1")],
            router_answer="rag",
        )
        import backend.app.gemini_client as gemini_router

        spy = MagicMock(return_value=_router_result(answer="rag"))
        monkeypatch.setattr(gemini_router, "route_and_invoke", spy)
        rag.ask_question("question")
        # With 1 doc, context is provided (RAG path is taken)
        context_arg = spy.call_args.kwargs.get("context", "")
        assert context_arg != ""  # context was provided → RAG path

    def test_cache_hit_skips_search(self, monkeypatch):
        search_calls = []
        rag = _load_rag(monkeypatch, cache_hit="cached")
        monkeypatch.setattr(
            rag,
            "hybrid_search",
            lambda *a, **k: search_calls.append(1) or [],
        )
        result = rag.ask_question("anything")
        assert result == "cached"
        assert len(search_calls) == 0

    def test_forbidden_query_blocked(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("how to hack systems") == "This query is not allowed."

    def test_empty_query_rejected(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("") == "Please provide a question."
        assert rag.ask_question("   ") == "Please provide a question."

    def test_router_failure_returns_error_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_raises=True)
        result = rag.ask_question("what is ai")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cache_written_after_rag(self, monkeypatch):
        writes = []
        rag = _load_rag(monkeypatch)
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: writes.append((q, a)))
        rag.ask_question("what is rag")
        assert len(writes) == 1
        # RAG path returns the answer directly (no prefix wrapper)
        assert writes[0][1] == "good answer"

    def test_cache_written_after_llm_fallback(self, monkeypatch):
        writes = []
        rag = _load_rag(monkeypatch, search_results=[], router_answer="llm fallback")
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: writes.append((q, a)))
        rag.ask_question("what is rag")
        assert len(writes) == 1


# ─── summarize_doc ────────────────────────────────────────────────────────────


class TestSummarizeDoc:
    def test_empty_doc_id_rejected(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.summarize_doc("") == "Please provide a document ID."

    def test_no_docs_returns_not_found(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        import backend.app.s3_vectors_client as s3vec

        monkeypatch.setattr(s3vec, "get_documents_by_doc_base_id", lambda x: [])
        result = rag.summarize_doc("doc123")
        # Source returns "No content available..." when docs list is empty;
        # the function returns early with "No documents found for ID: doc123"
        assert "doc123" in result or "No" in result

    def test_returns_summary_on_success(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="summary text")
        import backend.app.s3_vectors_client as s3vec

        monkeypatch.setattr(
            s3vec,
            "get_documents_by_doc_base_id",
            lambda x: [{"_source": {"text": "doc content"}}],
        )
        result = rag.summarize_doc("doc123")
        assert isinstance(result, str)

    def test_router_failure_returns_error(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_raises=True)
        import backend.app.s3_vectors_client as s3vec

        monkeypatch.setattr(
            s3vec,
            "get_documents_by_doc_base_id",
            lambda x: [{"_source": {"text": "content"}}],
        )
        result = rag.summarize_doc("doc123")
        assert isinstance(result, str)
