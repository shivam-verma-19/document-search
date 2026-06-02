# backend/tests/_stubs.py

import sys
from unittest.mock import MagicMock


def install_all_stubs():
    # ── S3 Vectors ────────────────────────────────────────────────────────────
    mock_s3v = MagicMock()
    mock_s3v.search_similar = MagicMock(
        return_value=["mock document 1", "mock document 2"]
    )
    mock_s3v.search_documents = MagicMock(
        return_value=[
            {
                "_id": "id1",
                "_source": {
                    "text": "mock document 1",
                    "metadata": {"filename": "test.pdf", "chunk_index": 0},
                },
                "score": 0.9,
            },
            {
                "_id": "id2",
                "_source": {
                    "text": "mock document 2",
                    "metadata": {"filename": "test.pdf", "chunk_index": 1},
                },
                "score": 0.8,
            },
        ]
    )
    mock_s3v.index_document = MagicMock()
    mock_s3v.get_all_documents = MagicMock(
        return_value=["mock document 1", "mock document 2"]
    )
    mock_s3v.delete_document = MagicMock(return_value={"result": "deleted"})
    sys.modules["backend.app.s3_vectors_client"] = mock_s3v
    sys.modules["backend.app.faiss_client"] = mock_s3v  # legacy alias

    # ── Gemini client ─────────────────────────────────────────────────────────
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
    mock_gemini.GEMINI_MODEL = "gemini-2.5-flash"

    class _ModelResponse:
        def __init__(self, model, text, success, confidence=0.0, error=None):
            self.model = model
            self.text = text
            self.success = success
            self.confidence = confidence
            self.error = error

    class _ClassifierResult:
        def __init__(self, complexity="simple", score=0.1, signals=None):
            self.complexity = complexity
            self.score = score
            self.signals = signals or []

    mock_gemini.ModelResponse = _ModelResponse
    mock_gemini.ClassifierResult = _ClassifierResult
    mock_gemini.classify_complexity = MagicMock(return_value=_ClassifierResult())
    mock_gemini.score_confidence = MagicMock(return_value=0.80)

    sys.modules["backend.app.gemini_client"] = mock_gemini
    sys.modules["backend.app.bedrock_router"] = mock_gemini  # legacy alias

    # ── Embeddings ────────────────────────────────────────────────────────────
    mock_embeddings = MagicMock()
    mock_embeddings.get_embedding = MagicMock(return_value=[0.1] * 768)
    sys.modules["backend.app.embeddings"] = mock_embeddings

    # ── Reranker — passthrough by default ────────────────────────────────────
    mock_reranker = MagicMock()
    mock_reranker.rerank = MagicMock(side_effect=lambda q, docs, **kw: docs)
    sys.modules["backend.app.reranker"] = mock_reranker

    # ── Query expansion — passthrough by default ──────────────────────────────
    mock_qe = MagicMock()
    mock_qe.generate_hyde_query = MagicMock(side_effect=lambda q: q)
    mock_qe.HYDE_ENABLED = True
    sys.modules["backend.app.query_expansion"] = mock_qe

    # ── Eval — no-op by default ───────────────────────────────────────────────
    mock_eval = MagicMock()
    mock_eval.evaluate_retrieval = MagicMock()
    mock_eval.evaluate_answer = MagicMock()
    mock_eval.log_eval = MagicMock()
    mock_eval.get_eval_records = MagicMock(return_value=[])
    sys.modules["backend.app.eval"] = mock_eval


def install_extra_stubs():
    """Call after install_all_stubs() for tests that touch BM25/secrets."""
    mock_bm25_cache = MagicMock()
    mock_bm25_cache.get_corpus = MagicMock(return_value=["doc 1", "doc 2", "doc 3"])
    mock_bm25_cache.set_corpus = MagicMock()
    mock_bm25_cache.append_to_corpus = MagicMock()
    mock_bm25_cache._warm_version = 0
    sys.modules["backend.app.bm25_cache"] = mock_bm25_cache

    mock_secrets = MagicMock()
    mock_secrets.get_secret = MagicMock(return_value="test-api-key")
    sys.modules["backend.app.secrets"] = mock_secrets
