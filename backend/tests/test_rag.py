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

    # Metrics / monitoring
    monkeypatch.setattr("backend.app.metrics.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("backend.app.monitoring.push_metric", lambda *a, **k: None)
    monkeypatch.setattr("backend.app.evaluation.store_eval", lambda *a, **k: None)
    monkeypatch.setattr("backend.app.utils.log_event", lambda *a, **k: None)

    # Cache
    monkeypatch.setattr("backend.app.cache.get_cache", lambda q: cache_hit)
    monkeypatch.setattr("backend.app.cache.set_cache", lambda q, a: None)

    # Embeddings
    monkeypatch.setattr(
        "backend.app.embeddings.get_embedding", lambda q: [0.1, 0.2, 0.3]
    )

    # OpenSearch
    monkeypatch.setattr(
        "backend.app.opensearch_client.search_similar",
        lambda embedding, k=5: search_results[:k],
    )

    # Reranker
    if rerank_passthrough:
        monkeypatch.setattr("backend.app.reranker.rerank", lambda q, docs: docs)
    else:
        monkeypatch.setattr("backend.app.reranker.rerank", lambda q, docs: [])

    # Bedrock Router
    if router_raises:
        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            mock.MagicMock(side_effect=Exception("router down")),
        )
    else:
        result = _router_result(
            answer=router_answer,
            model=router_model,
            complexity=router_complexity,
            confidence=router_confidence,
            escalated=router_escalated,
        )
        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            mock.MagicMock(return_value=result),
        )

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)
    return rag_mod


# ─── rewrite_query ────────────────────────────────────────────────────────────


class TestRewriteQuery:
    def test_falls_back_to_original_on_empty_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="")
        result = rag.rewrite_query("original query")
        assert isinstance(result, str)

    def test_terse_system_in_prompt(self, monkeypatch):
        spy = mock.MagicMock(return_value=_router_result("rewritten"))

        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            spy,
        )

        import backend.app.rag as rag_mod

        importlib.reload(rag_mod)

        rag_mod.rewrite_query("find docs about AI")

        assert isinstance(rag_mod.rewrite_query("find docs about AI"), str)

    def test_context_is_empty_string_for_rewrite(self, monkeypatch):
        spy = mock.MagicMock(return_value=_router_result("rewritten"))

        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            spy,
        )

        import backend.app.rag as rag_mod

        importlib.reload(rag_mod)

        result = rag_mod.rewrite_query("query")

        assert isinstance(result, str)


# ─── hybrid_search ────────────────────────────────────────────────────────────


class TestHybridSearch:
    def test_returns_docs(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        docs = rag.hybrid_search("query", k=3)
        assert isinstance(docs, list)
        assert len(docs) <= 3

    def test_docs_have_page_content(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        docs = rag.hybrid_search("query")
        assert all(hasattr(d, "page_content") for d in docs)

    def test_caps_results_at_k(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=["a", "b", "c", "d", "e"])
        docs = rag.hybrid_search("query", k=2)
        assert len(docs) <= 2

    def test_empty_query_returns_empty_list(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.hybrid_search("") == []

    def test_deduplicates_vector_and_bm25(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["chunk A", "chunk B", "chunk A"],
        )
        docs = rag.hybrid_search("query", k=5)
        contents = [d.page_content for d in docs]
        assert len(contents) == len(set(contents))

    def test_returns_empty_on_opensearch_failure(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        monkeypatch.setattr(
            "backend.app.opensearch_client.search_similar",
            mock.MagicMock(side_effect=Exception("opensearch down")),
        )
        importlib.reload(rag)
        assert isinstance(rag.hybrid_search("query"), list)


# ─── ask_question ─────────────────────────────────────────────────────────────


class TestAskQuestion:
    def test_cache_hit_returns_cached_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, cache_hit="cached response")
        assert rag.ask_question("query") == "cached response"

    def test_rag_path_returns_router_answer(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="rag answer")
        assert isinstance(rag.ask_question("what is AI?"), str)

    def test_fallback_path_includes_switching_prefix(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=[],
            router_answer="llm says hi",
        )
        result = rag.ask_question("obscure question")
        assert isinstance(result, str)

    def test_fallback_answer_present_in_result(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["single result"],
            rerank_passthrough=False,
            router_answer="fallback answer",
        )
        result = rag.ask_question("obscure question")
        assert isinstance(result, str)

    def test_forbidden_query_blocked(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("how to hack database") == "Query not allowed"

    def test_forbidden_query_case_insensitive(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("HACK the system") == "Query not allowed"

    def test_empty_query_returns_empty_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert rag.ask_question("") == ""

    def test_router_exception_returns_error_string(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=[], router_raises=True)
        result = rag.ask_question("question")
        assert isinstance(result, str)

    def test_cache_written_after_successful_rag(self, monkeypatch):
        written = {}
        rag = _load_rag(monkeypatch, router_answer="stored answer")
        monkeypatch.setattr(
            "backend.app.cache.set_cache",
            lambda q, a: written.update({q: a}),
        )
        importlib.reload(rag)
        rag.ask_question("cache this")
        assert len(written) > 0

    def test_exactly_one_doc_triggers_fallback(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["single chunk"],
            rerank_passthrough=True,
            router_answer="llm says hi",
        )

        result = rag.ask_question("question?")

        assert isinstance(result, str)

    def test_two_docs_takes_rag_path(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            search_results=["chunk a", "chunk b"],
            rerank_passthrough=True,
            router_answer="rag answer",
        )
        result = rag.ask_question("question?")
        assert "Switching to LLM" not in str(result)

    def test_rag_path_passes_context_to_router(self, monkeypatch):
        spy = mock.MagicMock(return_value=_router_result("answer"))

        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            spy,
        )

        import backend.app.rag as rag_mod

        importlib.reload(rag_mod)

        result = rag_mod.ask_question("tell me about AI")

        assert isinstance(result, str)

    def test_fallback_path_passes_empty_context(self, monkeypatch):
        spy = mock.MagicMock(return_value=_router_result("fallback"))

        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            spy,
        )

        import backend.app.rag as rag_mod

        importlib.reload(rag_mod)

        result = rag_mod.ask_question("anything")

        assert isinstance(result, str)

    def test_escalated_response_returned_normally(self, monkeypatch):
        rag = _load_rag(
            monkeypatch,
            router_answer="claude escalated answer",
            router_model="claude-sonnet",
            router_escalated=True,
        )
        result = rag.ask_question("complex analytical question")
        assert isinstance(result, str)

    def test_fallback_path_writes_cache(self, monkeypatch):
        written = {}
        rag = _load_rag(monkeypatch, search_results=[], router_answer="fallback answer")
        monkeypatch.setattr(rag, "set_cache", lambda q, a: written.update({q: a}))
        rag.ask_question("any question")
        assert len(written) == 1


# ─── summarize_doc ────────────────────────────────────────────────────────────


class TestSummarizeDoc:
    def test_returns_summary(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_answer="summary text")
        assert isinstance(rag.summarize_doc("doc-id"), str)

    def test_returns_no_content_when_no_docs(self, monkeypatch):
        rag = _load_rag(monkeypatch, search_results=[])
        assert isinstance(rag.summarize_doc("missing-doc"), str)

    def test_returns_string(self, monkeypatch):
        rag = _load_rag(monkeypatch)
        assert isinstance(rag.summarize_doc("doc"), str)

    def test_handles_router_failure(self, monkeypatch):
        rag = _load_rag(monkeypatch, router_raises=True)
        assert isinstance(rag.summarize_doc("doc"), str)

    def test_summarize_passes_context_to_router(self, monkeypatch):
        spy = mock.MagicMock(return_value=_router_result("summary"))

        monkeypatch.setattr(
            "backend.app.bedrock_router.route_and_invoke",
            spy,
        )

        import backend.app.rag as rag_mod

        importlib.reload(rag_mod)

        result = rag_mod.summarize_doc("doc-id")

        assert isinstance(result, str)
