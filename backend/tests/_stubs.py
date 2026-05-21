# backend/tests/_stubs.py

import sys
from unittest.mock import MagicMock


def install_all_stubs():
    # ── FAISS ─────────────────────────────────────────────────────────────
    mock_faiss = MagicMock()
    mock_faiss.search_similar = MagicMock(
        return_value=["mock document 1", "mock document 2", "mock document 3"]
    )
    mock_faiss.index_document = MagicMock()
    sys.modules["backend.app.faiss_client"] = mock_faiss

    # ── Bedrock router ─────────────────────────────────────────────────────────
    # Default result — individual tests override via monkeypatch.
    _default_result = {
        "answer": "mocked llm response",
        "model_used": "llama3-bedrock",
        "complexity": "simple",
        "confidence": 0.80,
        "escalated": False,
        "attempted": ["llama3-bedrock"],
        "errors": {},
    }

    mock_router = MagicMock()
    mock_router.route_and_invoke = MagicMock(return_value=_default_result)
    mock_router.CONFIDENCE_THRESHOLD = 0.65

    # Expose real dataclass-like objects so imports in test_bedrock_router work
    class _ModelResponse:
        def __init__(self, model, text, success, confidence=0.0, error=None):
            self.model = model
            self.text = text
            self.success = success
            self.confidence = confidence
            self.error = error

    mock_router.ModelResponse = _ModelResponse
    mock_router.classify_complexity = MagicMock()
    mock_router.score_confidence = MagicMock(return_value=0.80)

    sys.modules["backend.app.bedrock_router"] = mock_router

    # ── Embeddings (Bedrock Titan) ─────────────────────────────────────────────
    mock_embeddings = MagicMock()
    mock_embeddings.get_embedding = MagicMock(return_value=[0.1] * 1536)
    sys.modules["backend.app.embeddings"] = mock_embeddings
