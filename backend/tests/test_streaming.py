"""Tests for ask_question_stream — SSE streaming generator."""

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _evict_stubs():
    """Remove any MagicMock stubs installed by test_rag so real modules load."""
    for mod in ["backend.app.rag", "backend.app.reranker", "backend.app.gemini_client"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.rag", "backend.app.reranker", "backend.app.gemini_client"]:
        sys.modules.pop(mod, None)


from unittest.mock import MagicMock, patch

import pytest


def _collect_stream(gen) -> list[str]:
    return list(gen)


def _make_doc(text="chunk", doc_id="id1"):
    from backend.app.document_repository import SearchDocument

    return SearchDocument(page_content=text, doc_id=doc_id)


class TestAskQuestionStream:
    def test_yields_sse_format(self, monkeypatch):
        import backend.app.cache as cache_mod
        import backend.app.metrics as metrics_mod
        import backend.app.rag as rag_mod

        monkeypatch.setattr(cache_mod, "get_cache", lambda q: None)
        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: None)
        monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)
        monkeypatch.setattr(rag_mod, "hybrid_search", lambda q, **kw: [_make_doc()])
        monkeypatch.setattr(rag_mod, "rerank_documents", lambda q, d: d)
        monkeypatch.setattr(rag_mod, "_run_eval_async", lambda *a, **k: None)

        mock_chunk = MagicMock(text="token")
        # _get_client is imported inside ask_question_stream from gemini_client
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content_stream.return_value = iter(
                [mock_chunk]
            )
            chunks = _collect_stream(rag_mod.ask_question_stream("test question"))

        assert all(c.startswith("data: ") for c in chunks)

    def test_ends_with_done_sentinel(self, monkeypatch):
        import backend.app.cache as cache_mod
        import backend.app.metrics as metrics_mod
        import backend.app.rag as rag_mod

        monkeypatch.setattr(cache_mod, "get_cache", lambda q: None)
        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: None)
        monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)
        monkeypatch.setattr(rag_mod, "hybrid_search", lambda q, **kw: [_make_doc()])
        monkeypatch.setattr(rag_mod, "rerank_documents", lambda q, d: d)
        monkeypatch.setattr(rag_mod, "_run_eval_async", lambda *a, **k: None)

        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content_stream.return_value = iter(
                [MagicMock(text="hi")]
            )
            chunks = _collect_stream(rag_mod.ask_question_stream("question"))

        assert chunks[-1] == "data: [DONE]\n\n"

    def test_empty_query_yields_error_and_done(self, monkeypatch):
        from backend.app.rag import ask_question_stream

        chunks = _collect_stream(ask_question_stream(""))
        assert any("[DONE]" in c for c in chunks)
        assert len(chunks) == 2  # error message + DONE

    def test_forbidden_query_yields_blocked_and_done(self, monkeypatch):
        from backend.app.rag import ask_question_stream

        chunks = _collect_stream(ask_question_stream("how to hack"))
        assert any("not allowed" in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_cached_answer_streamed_token_by_token(self, monkeypatch):
        import backend.app.cache as cache_mod

        monkeypatch.setattr(cache_mod, "get_cache", lambda q: "cached answer text")
        from backend.app.rag import ask_question_stream

        chunks = _collect_stream(ask_question_stream("question"))
        full_text = "".join(
            c.replace("data: ", "").replace("\n\n", "")
            for c in chunks
            if "[DONE]" not in c
        )
        assert "cached" in full_text
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_gemini_failure_yields_error_and_done(self, monkeypatch):
        import backend.app.gemini_client as gc
        import backend.app.metrics as metrics_mod
        import backend.app.rag as rag_mod

        gc._client = None
        monkeypatch.setattr(rag_mod, "get_cached_answer", lambda q: None)
        monkeypatch.setattr(rag_mod, "set_cached_answer", lambda q, a: None)
        monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)
        monkeypatch.setattr(rag_mod, "hybrid_search", lambda q, **kw: [])
        monkeypatch.setattr(rag_mod, "_run_eval_async", lambda *a, **k: None)

        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content_stream.side_effect = (
                Exception("api down")
            )
            chunks = _collect_stream(rag_mod.ask_question_stream("question"))

        assert chunks[-1] == "data: [DONE]\n\n"
        assert any("trouble" in c for c in chunks)

    def test_newlines_escaped_for_sse(self, monkeypatch):
        """Newlines in tokens must be escaped as \\n for SSE transport."""
        import backend.app.cache as cache_mod
        import backend.app.metrics as metrics_mod
        import backend.app.rag as rag_mod

        monkeypatch.setattr(cache_mod, "get_cache", lambda q: None)
        monkeypatch.setattr(cache_mod, "set_cache", lambda q, a: None)
        monkeypatch.setattr(metrics_mod, "log_metrics", lambda *a, **k: None)
        monkeypatch.setattr(rag_mod, "hybrid_search", lambda q, **kw: [])
        monkeypatch.setattr(rag_mod, "_run_eval_async", lambda *a, **k: None)

        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content_stream.return_value = iter(
                [MagicMock(text="line1\nline2")]
            )
            chunks = _collect_stream(rag_mod.ask_question_stream("q"))

        token_chunks = [c for c in chunks if "[DONE]" not in c]
        # Each SSE frame ends with \n\n (terminator); any embedded newline in the
        # token itself must be escaped to \\n so it doesn't break the SSE framing.
        # Verify the raw \n from the token was escaped: the payload (between "data: "
        # and the final "\n\n") must not contain a bare \n.
        for c in token_chunks:
            payload = c[len("data: ") : -2]  # strip "data: " prefix and "\n\n" suffix
            assert "\n" not in payload, f"Bare newline found in SSE payload: {c!r}"
