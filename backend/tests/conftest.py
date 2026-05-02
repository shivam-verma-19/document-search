"""
Shared pytest fixtures for the document-search test suite.

All AWS calls are intercepted by moto so no real credentials are needed.
LLM / embedding / vector-db calls are replaced by lightweight stubs so the
tests run offline and deterministically.
"""

import io
import json
import os
import types
import uuid

import boto3
import pytest
from moto import mock_aws

# ---------------------------------------------------------------------------
# Environment variables (must be set BEFORE any app module is imported)
# ---------------------------------------------------------------------------
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault(
    "QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
)
os.environ.setdefault("SECRET_NAME", "rag-secrets")
os.environ.setdefault("BUCKET_NAME", "rag-pipeline-upload-bucket")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str = "hello world"):
    """Return a minimal LangChain-style Document stub."""
    doc = types.SimpleNamespace()
    doc.page_content = text
    doc.metadata = {}
    return doc


# ---------------------------------------------------------------------------
# AWS infrastructure fixtures (moto)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def aws_credentials():
    """Fake credentials so moto never touches real AWS."""
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
def dynamodb_tables(aws_credentials):
    with mock_aws():
        db = boto3.resource("dynamodb", region_name="us-east-1")

        for table_name, key in [
            ("rag-cache", "query"),
            ("rag-metrics", "id"),
            ("rag-eval", "query"),
        ]:
            db.create_table(
                TableName=table_name,
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
        yield db


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
            SecretString=json.dumps(
                {"OPENAI_API_KEY": "sk-test", "PINECONE_API_KEY": "pc-test"}
            ),
        )
        yield sm


# ---------------------------------------------------------------------------
# Lightweight document stub
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_doc():
    return _make_doc("The quarterly revenue increased by 12% year over year.")


@pytest.fixture()
def sample_docs():
    return [_make_doc(f"Document chunk number {i}.") for i in range(5)]


# ---------------------------------------------------------------------------
# Upload file helper
# ---------------------------------------------------------------------------


@pytest.fixture()
def upload_file():
    """Return a FastAPI-compatible UploadFile stub."""

    class FakeUploadFile:
        filename = "test.pdf"
        content_type = "application/pdf"
        file = io.BytesIO(b"fake pdf content for testing")

    return FakeUploadFile()
