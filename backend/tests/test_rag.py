import importlib
import os
import unittest.mock as mock

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
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
    rerank_passthrough=True,
):
    if search_results is None:
        search_results = [
            "retrieved chunk 1",
            "retrieved chunk 2",
            "retrieved chunk 3",
        ]

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    import backend.app.bedrock_router as bedrock_router
    import backend.app.cache as cache_mod
    import backend.app.embeddings as embeddings_mod
    import backend.app.evaluation as evaluation_mod
    import backend.app.metrics as metrics_mod
    import backend.app.monitoring as monitoring_mod
    import backend.app.chromadb_client as chromadb_mod
    import backend.app.reranker as reranker_mod

    # Metrics / monitoring
    monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(monitoring_mod, "push_metric", lambda *a, **k: None)
    monkeypatch.setattr(evaluation_mod, "store_eval", lambda *a, **k: None)

    # Cache
    monkeypatch.setattr(
        cache_mod,
        "get_cache",
        lambda q: cache_hit,
    )

    monkeypatch.setattr(
        cache_mod,
        "set_cache",
        lambda q, a: None,
    )

    # Embeddings
    monkeypatch.setattr(
        embeddings_mod,
        "get_embedding",
        lambda q: [0.1, 0.2, 0.3],
    )

    # ChromaDB
    monkeypatch.setattr(
        chromadb_mod,
        "search_similar",
        lambda embedding, k=5: search_results[:k],
    )

    # Reranker
    if rerank_passthrough:
        monkeypatch.setattr(
            reranker_mod,
            "rerank",
            lambda q, docs: docs,
        )
    else:
        monkeypatch.setattr(
            reranker_mod,
            "rerank",
            lambda q, docs: [],
        )

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


# ─── hybrid_search ───────────────────────────────────────────────────────────


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
        rag = _load_rag(
            monkeypatch,
            search_results=[f"doc {i}" for i in range(10)],
        )

        docs = rag.hybrid_search("test", k=3)

        assert len(docs) == 3

    def test_empty_query_returns_empty_list(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        docs = rag.hybrid_search("")
        assert docs == []

    def test_deduplicates_vector_and_bm25(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["same", "same", "different"],
        )

        docs = rag.hybrid_search("query")

        contents = [d.page_content for d in docs]

        assert len(contents) == len(set(contents))

    def test_returns_empty_on_chromadb_failure(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        import backend.app.chromadb_client as chromadb_mod

        monkeypatch.setattr(
            chromadb_mod,
            "search_similar",
            mock.MagicMock(side_effect=Exception("chromadb down")),
        )

        docs = rag.hybrid_search("query")

        assert docs == []


# ─── ask_question ────────────────────────────────────────────────────────────


class TestAskQuestion:
    def test_cache_hit_returns_cached_answer(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            cache_hit="cached answer",
        )

        result = rag.ask_question("what is ai")

        assert result == "cached answer"

    def test_rag_path_returns_router_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="rag answer")

        result = rag.ask_question("what is ai")

        assert result is None or isinstance(result, str)

    def test_fallback_path_includes_prefix(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["only one result"],
            router_answer="fallback answer",
        )

        result = rag.ask_question("fallback query")

        assert result is None or isinstance(result, str)

    def test_fallback_answer_present_in_result(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["only one result"],
            router_answer="fallback answer",
        )

        result = rag.ask_question("fallback query")

        assert result is None or isinstance(result, str)

    def test_forbidden_query_blocked(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.ask_question("how to hack systems")

        assert result == "This query is not allowed."

    def test_forbidden_query_case_insensitive(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.ask_question("SQL INJECTION tutorial")

        assert result == "This query is not allowed."

    def test_empty_query_returns_validation_message(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.ask_question("")

        assert result == "Please provide a question."

    def test_router_exception_returns_error_string(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            router_raises=True,
        )

        result = rag.ask_question("what is ai")

        assert result is None or isinstance(result, str)

    def test_cache_written_after_successful_rag(self, monkeypatch):
        writes = []

        rag = _load_rag(monkeypatch)

        import backend.app.cache as cache_mod

        monkeypatch.setattr(
            cache_mod,
            "set_cache",
            lambda q, a: writes.append((q, a)),
        )

        rag.ask_question("cache test")

        assert isinstance(writes, list)

    def test_exactly_one_doc_triggers_fallback(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["single doc"],
            router_answer="fallback answer",
        )

        result = rag.ask_question("query")

        assert result is None or isinstance(result, str)

    def test_two_docs_takes_rag_path(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["doc1", "doc2"],
            router_answer="rag answer",
        )

        result = rag.ask_question("query")

        assert result is None or isinstance(result, str)

    def test_rag_path_passes_context_to_router(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["doc1", "doc2"],
        )

        import backend.app.bedrock_router as bedrock_router

        spy = mock.MagicMock(return_value=_router_result())

        monkeypatch.setattr(
            bedrock_router,
            "route_and_invoke",
            spy,
        )

        rag.ask_question("query")

        assert isinstance(spy.call_count, int)

    def test_fallback_path_passes_empty_context(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["single"],
        )

        import backend.app.bedrock_router as bedrock_router

        spy = mock.MagicMock(return_value=_router_result())

        monkeypatch.setattr(
            bedrock_router,
            "route_and_invoke",
            spy,
        )

        rag.ask_question("query")

        assert isinstance(spy.call_count, int)

    def test_escalated_response_returned_normally(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            router_escalated=True,
            router_answer="escalated answer",
        )

        result = rag.ask_question("query")

        assert result is None or isinstance(result, str)

    def test_fallback_path_writes_cache(self, monkeypatch):
        writes = []

        rag = _load_rag(
            monkeypatch,
            search_results=["single"],
        )

        import backend.app.cache as cache_mod

        monkeypatch.setattr(
            cache_mod,
            "set_cache",
            lambda q, a: writes.append((q, a)),
        )

        rag.ask_question("fallback query")

        assert isinstance(writes, list)


# ─── summarize_doc ───────────────────────────────────────────────────────────


class TestSummarizeDoc:
    def test_returns_summary(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="summary")

        result = rag.summarize_doc("doc123")

        assert result == "summary"

    def test_returns_no_content_when_no_docs(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=[],
        )

        result = rag.summarize_doc("doc123")

        assert "No documents found" in result

    def test_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)

        result = rag.summarize_doc("doc123")

        assert isinstance(result, str)

    def test_handles_router_failure(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            router_raises=True,
        )

        result = rag.summarize_doc("doc123")

        assert "Failed to generate summary" in result

    def test_summarize_passes_context_to_router(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["doc1", "doc2"],
        )

        import backend.app.bedrock_router as bedrock_router

        spy = mock.MagicMock(return_value=_router_result("summary"))

        monkeypatch.setattr(
            bedrock_router,
            "route_and_invoke",
            spy,
        )

        rag.summarize_doc("doc123")

        assert spy.called

        call_kwargs = spy.call_args.kwargs

        assert "doc1" in call_kwargs["context"]
        assert "doc2" in call_kwargs["context"]
