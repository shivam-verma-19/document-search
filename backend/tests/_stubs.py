# backend/tests/_stubs.py

import sys
from unittest.mock import MagicMock


def install_all_stubs():
    """
    Install external dependency stubs BEFORE importing app code.
    This prevents real network/API calls in tests.
    """

    # =========================
    # 1. Mock ChatOpenAI (LLM)
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
    # 2. Mock Pinecone Vector Store
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
