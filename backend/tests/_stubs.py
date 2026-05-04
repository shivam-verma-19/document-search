# backend/tests/_stubs.py

import sys
import types
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
            return [MockDoc(f"doc content {i} for {query}") for i in range(min(k, 5))]

    mock_pinecone_module = MagicMock()
    setattr(mock_pinecone_module, "PineconeVectorStore", MockVectorStore)

    sys.modules["langchain_pinecone"] = mock_pinecone_module

    # =========================
    # 3. Mock jose (JWT)
    # =========================
    jose_module = types.ModuleType("jose")

    class MockJWT:
        @staticmethod
        def decode(*args, **kwargs):
            return {"sub": "user-id"}

    setattr(jose_module, "jwt", MockJWT)
    sys.modules["jose"] = jose_module

    # =========================
    # 4. Mock pydantic_settings
    # =========================
    pydantic_settings_module = types.ModuleType("pydantic_settings")

    class BaseSettings:
        def __init__(self, *args, **kwargs):
            pass

    setattr(pydantic_settings_module, "BaseSettings", BaseSettings)
    sys.modules["pydantic_settings"] = pydantic_settings_module

    # =========================
    # 5. Mock boto3 (S3 / SQS)
    # =========================
    boto3_module = types.ModuleType("boto3")

    class MockS3Client:
        def upload_fileobj(self, file, bucket, key):
            return None

    class MockSQSClient:
        def send_message(self, QueueUrl=None, MessageBody=None):
            return {"MessageId": "mocked-message-id"}

    def client(service_name, *args, **kwargs):
        if service_name == "s3":
            return MockS3Client()
        elif service_name == "sqs":
            return MockSQSClient()
        return MagicMock()

    setattr(boto3_module, "client", client)
    sys.modules["boto3"] = boto3_module

    # =========================
    # 6. Mock dotenv
    # =========================
    dotenv_module = types.ModuleType("dotenv")
    setattr(dotenv_module, "load_dotenv", lambda *a, **k: None)

    sys.modules["dotenv"] = dotenv_module
