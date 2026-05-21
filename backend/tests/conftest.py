"""
Shared pytest fixtures for the document-search test suite.

All AWS calls are intercepted by moto.
LLM / embeddings / ChromaDB are mocked.
Tests run fully offline.
"""

import importlib
import io
import json
import os
import types

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

# ---------------------------------------------------------------------------
# ENV VARS (before imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault(
    "QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
)
os.environ.setdefault("SECRET_NAME", "rag-secrets")
os.environ.setdefault("BUCKET_NAME", "rag-pipeline-upload-bucket")
os.environ.setdefault("ALLOWED_UPLOAD_EXTENSIONS", "pdf,txt,docx,doc")
os.environ.setdefault(
    "ALLOWED_UPLOAD_MIMES",
    "application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword",
)
os.environ.setdefault("FORBIDDEN_UPLOAD_PATTERNS", "")
os.environ.setdefault("CHROMA_PERSIST_DIR", "http:///tmp/chroma-test")
os.environ.setdefault("USE_BEDROCK", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_doc(text: str = "hello world"):
    doc = types.SimpleNamespace()
    doc.page_content = text
    doc.metadata = {}
    return doc


# ---------------------------------------------------------------------------
# AWS MOCKS (moto)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture()
def s3_bucket(aws_credentials):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-pipeline-upload-bucket")
        yield s3


@pytest.fixture()
def sqs_queue(aws_credentials):
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-east-1")
        resp = sqs.create_queue(QueueName="test-queue")
        os.environ["QUEUE_URL"] = resp["QueueUrl"]
        yield sqs, resp["QueueUrl"]


@pytest.fixture()
def secretsmanager_secret(aws_credentials):
    with mock_aws():
        sm = boto3.client("secretsmanager", region_name="us-east-1")
        sm.create_secret(
            Name="rag-secrets",
            SecretString=json.dumps({"OPENAI_API_KEY": "sk-test"}),
        )
        yield sm


# ---------------------------------------------------------------------------
# FILE MOCK
# ---------------------------------------------------------------------------
@pytest.fixture()
def upload_file():
    class FakeUploadFile:
        filename = "test.pdf"
        content_type = "application/pdf"
        file = io.BytesIO(b"fake pdf content")

    return FakeUploadFile()


# ---------------------------------------------------------------------------
# APP FIXTURE (CRITICAL)
# ---------------------------------------------------------------------------
@pytest.fixture()
def app(monkeypatch):
    """
    Build FastAPI app with ALL external dependencies mocked.
    """

    # -------------------------
    # Auth
    # -------------------------
    monkeypatch.setattr("backend.app.auth.verify_token", lambda token=None: "user-id")
    monkeypatch.setattr("backend.app.auth.verify_cognito_token", lambda: "user-id")
    monkeypatch.setattr("backend.app.auth.optional_auth", lambda: "user-id")

    # -------------------------
    # RAG (LLM)
    # -------------------------
    monkeypatch.setattr("backend.app.rag.ask_question", lambda q: "mock answer")
    monkeypatch.setattr("backend.app.rag.summarize_doc", lambda d: "mock summary")

    # -------------------------
    # 🔥 NEW: Embeddings
    # -------------------------
    monkeypatch.setattr(
        "backend.app.embeddings.get_embedding",
        lambda text: [0.1] * 1536,
    )

    # -------------------------
    # 🔥 NEW: ChromaDB
    # -------------------------
    monkeypatch.setattr(
        "backend.app.chromadb_client.index_document",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "backend.app.chromadb_client.search_similar",
        lambda emb, k=5: ["mock context 1", "mock context 2"],
    )

    # -------------------------
    # Ingest
    # -------------------------
    monkeypatch.setattr(
        "backend.app.ingest.upload_file_to_s3",
        lambda file, user: "mocked-key",
    )
    monkeypatch.setattr(
        "backend.app.ingest.enqueue_file",
        lambda key, user: None,
    )

    # -------------------------
    # Metrics
    # -------------------------
    monkeypatch.setattr(
        "backend.app.metrics.get_metrics",
        lambda: [{"id": "1", "latency": 100}],
    )

    import backend.app.main as main_mod

    importlib.reload(main_mod)

    return main_mod.app


# ---------------------------------------------------------------------------
# CLIENT FIXTURE
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
