# backend/tests/_stubs.py

import sys
from unittest.mock import MagicMock


def install_all_stubs():
    # ── S3 Vectors (replaces FAISS) ───────────────────────────────────────────
    mock_s3v = MagicMock()
    mock_s3v.search_similar = MagicMock(
        return_value=["mock document 1", "mock document 2", "mock document 3"]
    )
    mock_s3v.index_document = MagicMock()
    mock_s3v.get_all_documents = MagicMock(
        return_value=["mock document 1", "mock document 2", "mock document 3"]
    )
    sys.modules["backend.app.s3_vectors_client"] = mock_s3v
    # Keep faiss_client alias so any lingering import doesn't crash
    sys.modules["backend.app.faiss_client"] = mock_s3v

    # ── Gemini client (replaces bedrock_router) ───────────────────────────────
    _default_result = {
        "answer": "mocked llm response",
        "model_used": "gemini-2.5-flash",
        "complexity": "simple",
        "confidence": 0.80,
        "escalated": False,
        "attempted": ["gemini-2.5-flash"],
        "errors": {},
    }

    mock_gemini = MagicMock()
    mock_gemini.route_and_invoke = MagicMock(return_value=_default_result)
    mock_gemini.CONFIDENCE_THRESHOLD = 0.65

    class _ModelResponse:
        def __init__(self, model, text, success, confidence=0.0, error=None):
            self.model = model
            self.text = text
            self.success = success
            self.confidence = confidence
            self.error = error

    mock_gemini.ModelResponse = _ModelResponse
    mock_gemini.classify_complexity = MagicMock()
    mock_gemini.score_confidence = MagicMock(return_value=0.80)

    sys.modules["backend.app.gemini_client"] = mock_gemini
    # Keep bedrock_router alias so any lingering import doesn't crash
    sys.modules["backend.app.bedrock_router"] = mock_gemini

    # ── Embeddings (Gemini text-embedding-004) ─────────────────────────────────
    mock_embeddings = MagicMock()
    mock_embeddings.get_embedding = MagicMock(return_value=[0.1] * 768)
    sys.modules["backend.app.embeddings"] = mock_embeddings


def install_extra_stubs():
    """Call after install_all_stubs() for tests that touch new modules."""
    # ── BM25 corpus cache ─────────────────────────────────────────────────────
    mock_bm25_cache = MagicMock()
    mock_bm25_cache.get_corpus = MagicMock(return_value=["doc 1", "doc 2", "doc 3"])
    mock_bm25_cache.set_corpus = MagicMock()
    mock_bm25_cache.append_to_corpus = MagicMock()
    sys.modules["backend.app.bm25_cache"] = mock_bm25_cache

    # ── Secrets ───────────────────────────────────────────────────────────────
    mock_secrets = MagicMock()
    mock_secrets.get_secret = MagicMock(return_value="test-api-key")
    sys.modules["backend.app.secrets"] = mock_secrets
