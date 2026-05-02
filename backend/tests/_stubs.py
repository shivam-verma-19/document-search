# backend/tests/_stubs.py

import sys
from unittest.mock import MagicMock


def install_all_stubs():
    """
    Install all external dependency stubs BEFORE importing app code.
    This prevents real network/API calls in tests.
    """

    # =========================
    # 1. Mock boto3 (AWS Secrets Manager)
    # =========================
    import boto3

    mock_sm = MagicMock()
    mock_sm.get_secret_value.return_value = {
        "SecretString": '{"OPENAI_API_KEY": "test-key"}'
    }

    boto3.client = MagicMock(return_value=mock_sm)

    # =========================
    # 2. Mock ChatOpenAI (LLM)
    # =========================
    class MockLLM:
        def invoke(self, prompt):
            # Always return predictable response
            return MagicMock(content="mocked llm response")

    mock_openai_module = MagicMock()
    mock_openai_module.ChatOpenAI = MagicMock(return_value=MockLLM())
    mock_openai_module.OpenAIEmbeddings = MagicMock()

    sys.modules["langchain_openai"] = mock_openai_module

    # =========================
    # 3. Mock Pinecone Vector Store
    # =========================
    class MockDoc:
        def __init__(self, content):
            self.page_content = content

    class MockVectorStore:
        def __init__(self, *args, **kwargs):
            pass

        def similarity_search(self, query, k=5):
            # Return deterministic fake docs
            return [MockDoc(f"doc content {i} for {query}") for i in range(min(k, 5))]

    mock_pinecone_module = MagicMock()
    mock_pinecone_module.PineconeVectorStore = MockVectorStore

    sys.modules["langchain_pinecone"] = mock_pinecone_module

    # =========================
    # 4. Optional: Stub reranker (if heavy)
    # =========================
    sys.modules["backend.app.reranker"] = MagicMock(rerank=lambda query, docs: docs)

    # =========================
    # 5. Optional: Stub monitoring/metrics (no-op)
    # =========================
    sys.modules["backend.app.monitoring"] = MagicMock(
        push_metric=lambda *args, **kwargs: None
    )

    sys.modules["backend.app.metrics"] = MagicMock(
        log_metrics=lambda *args, **kwargs: None
    )

    sys.modules["backend.app.evaluation"] = MagicMock(
        store_eval=lambda *args, **kwargs: None
    )

    sys.modules["backend.app.utils"] = MagicMock(
        build_prompt=lambda context, query: f"{context}\n{query}",
        get_secrets=lambda: {},
        log_event=lambda *args, **kwargs: None,
    )

    # =========================
    # 6. Cache (simple in-memory)
    # =========================
    _cache = {}

    def get_cache(key):
        return _cache.get(key)

    def set_cache(key, value):
        _cache[key] = value

    sys.modules["backend.app.cache"] = MagicMock(
        get_cache=get_cache,
        set_cache=set_cache,
    )
