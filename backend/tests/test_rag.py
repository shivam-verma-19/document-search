import importlib
import os
import unittest.mock as mock

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")

from . import _stubs

_stubs.install_all_stubs()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _router_result(
    answer="good answer",
    model="llama3-bedrock",
    complexity="simple",
    confidence=0.80,
    escalated=False,
):
    return {
        "answer": answer,
        "model_used": model,
        "complexity": complexity,
        "confidence": confidence,
        "escalated": escalated,
        "attempted": [model],
        "errors": {},
    }


def _load_rag(
    monkeypatch,
    *,
    cache_hit=None,
    router_answer="good answer",
    router_model="llama3-bedrock",
    router_complexity="simple",
    router_confidence=0.80,
    router_escalated=False,
    router_raises=False,
    search_results=None,
    bm25_results=None,
    rerank_passthrough=True,
):
    if search_results is None:
        search_results = ["retrieved chunk 1", "retrieved chunk 2", "retrieved chunk 3"]

    if bm25_results is None:
        bm25_results = []

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    import backend.app.bedrock_router as bedrock_router
    import backend.app.cache as cache_mod
    import backend.app.embeddings as embeddings_mod
    import backend.app.faiss_client as faiss_mod
    import backend.app.metrics as metrics_mod
    import backend.app.reranker as reranker_mod
    from backend.app.hybrid import BM25Retriever

    # Metrics
    monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)

    # Cache
    monkeypatch.setattr(cache_mod, "get_cache", lambda q: cache_hit)
    monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: None)

    # Embeddings
    monkeypatch.setattr(embeddings_mod, "get_embedding", lambda q: [0.1, 0.2, 0.3])

    # FAISS — vector search results
    monkeypatch.setattr(
        faiss_mod, "search_similar", lambda embedding, k=5: search_results[:k]
    )

    # FAISS — all docs for BM25 corpus
    monkeypatch.setattr(
        faiss_mod, "get_all_documents", lambda: [r for r in search_results]
    )

    # BM25 — return bm25_results (default empty so RRF is driven by vector only)
    monkeypatch.setattr(
        BM25Retriever,
        "search",
        lambda self, q, k=5: [
            rag_mod.SearchDocument(page_content=t) for t in bm25_results[:k]
        ],
    )

    # Reranker
    if rerank_passthrough:
        monkeypatch.setattr(reranker_mod, "rerank", lambda q, docs: docs)
    else:
        monkeypatch.setattr(reranker_mod, "rerank", lambda q, docs: [])

    # Router
    if router_raises:
        monkeypatch.setattr(
            bedrock_router,
            "route_and_invoke",
            mock.MagicMock(side_effect=Exception("router down")),
        )
    else:
        monkeypatch.setattr(
            bedrock_router,
            "route_and_invoke",
            mock.MagicMock(
                return_value=_router_result(
                    answer=router_answer,
                    model=router_model,
                    complexity=router_complexity,
                    confidence=router_confidence,
                    escalated=router_escalated,
                )
            ),
        )

    return rag_mod


# ─── RRF ─────────────────────────────────────────────────────────────────────


class TestRRFFusion:
    def test_rrf_deduplicates_overlapping_docs(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        doc = rag.SearchDocument
        list1 = [doc("a"), doc("b"), doc("c")]
        list2 = [doc("b"), doc("c"), doc("d")]
        result = rag._reciprocal_rank_fusion(list1, list2, k=5)
        contents = [d.page_content for d in result]
        assert len(contents) == len(set(contents))

    def test_rrf_doc_in_both_lists_ranked_higher(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        doc = rag.SearchDocument
        # "shared" appears in both lists at rank 1 — should score highest
        list1 = [doc("shared"), doc("only_vector")]
        list2 = [doc("shared"), doc("only_bm25")]
        result = rag._reciprocal_rank_fusion(list1, list2, k=3)
        assert result[0].page_content == "shared"

    def test_rrf_empty_lists_returns_empty(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag._reciprocal_rank_fusion([], [], k=5) == []

    def test_rrf_one_empty_list_uses_other(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        doc = rag.SearchDocument
        result = rag._reciprocal_rank_fusion([doc("a"), doc("b")], [], k=5)
        assert len(result) == 2

    def test_rrf_respects_k_limit(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        doc = rag.SearchDocument
        list1 = [doc(f"v{i}") for i in range(10)]
        list2 = [doc(f"b{i}") for i in range(10)]
        result = rag._reciprocal_rank_fusion(list1, list2, k=3)
        assert len(result) == 3


# ─── hybrid_search ────────────────────────────────────────────────────────────


class TestHybridSearch:
    def test_returns_docs(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        docs = rag.hybrid_search("machine learning")
        assert len(docs) > 0

    def test_docs_have_page_content(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        docs = rag.hybrid_search("machine learning")
        assert hasattr(docs[0], "page_content")

    def test_caps_results_at_k(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=[f"doc {i}" for i in range(10)])
        docs = rag.hybrid_search("test", k=3)
        assert len(docs) <= 3

    def test_empty_query_returns_empty_list(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        docs = rag.hybrid_search("")
        assert docs == []

    def test_deduplicates_across_vector_and_bm25(self, monkeypatch):
        # same doc in both vector and BM25 — RRF should merge, not duplicate
        rag = _load_rag(
            monkeypatch,
            search_results=["same", "different_vector"],
            bm25_results=["same", "different_bm25"],
        )
        docs = rag.hybrid_search("query")
        contents = [d.page_content for d in docs]
        assert len(contents) == len(set(contents))

    def test_bm25_only_docs_included_when_vector_fails(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=[], bm25_results=["bm25_doc"])
        docs = rag.hybrid_search("query")
        assert any(d.page_content == "bm25_doc" for d in docs)

    def test_returns_empty_on_both_legs_failing(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        import backend.app.faiss_client as faiss_mod
        from backend.app.hybrid import BM25Retriever

        monkeypatch.setattr(
            faiss_mod,
            "search_similar",
            mock.MagicMock(side_effect=Exception("faiss down")),
        )
        monkeypatch.setattr(
            BM25Retriever, "search", mock.MagicMock(side_effect=Exception("bm25 down"))
        )
        docs = rag.hybrid_search("query")
        assert docs == []


# ─── get_cached_answer (cache bug fix) ───────────────────────────────────────


class TestCacheBugFix:
    def test_cache_miss_returns_none_not_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, cache_hit=None)
        result = rag.get_cached_answer("anything")
        # BUG FIX: must be None, not the string "None"
        assert result is None

    def test_cache_hit_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, cache_hit="cached value")
        result = rag.get_cached_answer("anything")
        assert result == "cached value"

    def test_cache_miss_does_not_short_circuit_pipeline(self, monkeypatch):
        # If cache returns None, ask_question should proceed to search
        rag = _load_rag(monkeypatch, cache_hit=None, router_answer="real answer")
        result = rag.ask_question("what is rag")
        assert result == "real answer"


# ─── ask_question ─────────────────────────────────────────────────────────────


class TestAskQuestion:
    def test_cache_hit_returns_cached_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, cache_hit="cached answer")
        result = rag.ask_question("what is ai")
        assert result == "cached answer"

    def test_rag_path_with_two_docs(self, monkeypatch):
        rag = _load_rag(
            monkeypatch, search_results=["doc1", "doc2"], router_answer="rag answer"
        )
        result = rag.ask_question("what is ai")
        assert result == "rag answer"

    def test_fallback_path_with_one_doc_includes_prefix(self, monkeypatch):
        rag = _load_rag(
            monkeypatch, search_results=["only one"], router_answer="fallback answer"
        )
        result = rag.ask_question("fallback query")
        assert "fallback answer" in result
        assert "couldn't find relevant documents" in result

    def test_forbidden_query_blocked(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("how to hack systems") == "This query is not allowed."

    def test_forbidden_query_case_insensitive(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert (
            rag.ask_question("SQL INJECTION tutorial") == "This query is not allowed."
        )

    def test_empty_query_returns_validation_message(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("") == "Please provide a question."

    def test_whitespace_only_query_blocked(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("   ") == "Please provide a question."

    def test_cache_written_after_successful_rag(self, monkeypatch):
        writes = []
        rag = _load_rag(monkeypatch, search_results=["doc1", "doc2"])
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: writes.append((q, a)))
        rag.ask_question("cache test")
        assert len(writes) == 1
        assert writes[0][1] == "good answer"

    def test_exactly_two_docs_takes_rag_path(self, monkeypatch):
        rag = _load_rag(
            monkeypatch, search_results=["doc1", "doc2"], router_answer="rag answer"
        )
        result = rag.ask_question("query")
        assert result == "rag answer"

    def test_exactly_one_doc_triggers_fallback(self, monkeypatch):
        rag = _load_rag(
            monkeypatch, search_results=["single doc"], router_answer="fallback answer"
        )
        result = rag.ask_question("query")
        assert "fallback answer" in result

    def test_router_exception_returns_error_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_raises=True)
        result = rag.ask_question("what is ai")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_escalated_response_returned_normally(self, monkeypatch):
        rag = _load_rag(
            monkeypatch, router_escalated=True, router_answer="escalated answer"
        )
        result = rag.ask_question("query")
        assert result == "escalated answer"

    def test_rag_path_passes_context_to_router(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=["doc1", "doc2"])
        import backend.app.bedrock_router as bedrock_router

        spy = mock.MagicMock(return_value=_router_result())
        monkeypatch.setattr(bedrock_router, "route_and_invoke", spy)
        rag.ask_question("query")
        assert spy.called
        kwargs = spy.call_args.kwargs
        assert "doc1" in kwargs["context"]
        assert "doc2" in kwargs["context"]

    def test_fallback_path_passes_empty_context(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=["single"])
        import backend.app.bedrock_router as bedrock_router

        spy = mock.MagicMock(return_value=_router_result())
        monkeypatch.setattr(bedrock_router, "route_and_invoke", spy)
        rag.ask_question("query")
        assert spy.called
        kwargs = spy.call_args.kwargs
        assert kwargs["context"] == ""

    def test_fallback_path_writes_cache(self, monkeypatch):
        writes = []
        rag = _load_rag(monkeypatch, search_results=["single"])
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: writes.append((q, a)))
        rag.ask_question("fallback query")
        assert len(writes) == 1


# ─── summarize_doc ────────────────────────────────────────────────────────────


class TestSummarizeDoc:
    def test_returns_summary(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="summary")
        assert rag.summarize_doc("doc123") == "summary"

    def test_returns_no_content_when_no_docs(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=[])
        assert "No documents found" in rag.summarize_doc("doc123")

    def test_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert isinstance(rag.summarize_doc("doc123"), str)

    def test_handles_router_failure(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_raises=True)
        assert "Failed to generate summary" in rag.summarize_doc("doc123")

    def test_summarize_passes_context_to_router(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=["doc1", "doc2"])
        import backend.app.bedrock_router as bedrock_router

        spy = mock.MagicMock(return_value=_router_result("summary"))
        monkeypatch.setattr(bedrock_router, "route_and_invoke", spy)
        rag.summarize_doc("doc123")
        assert spy.called
        kwargs = spy.call_args.kwargs
        assert "doc1" in kwargs["context"]
        assert "doc2" in kwargs["context"]

    def test_empty_doc_id_returns_validation_message(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.summarize_doc("") == "Please provide a document ID."
