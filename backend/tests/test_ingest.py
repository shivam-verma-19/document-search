"""
Tests for backend/app/ingest.py

`unstructured` is a heavy optional dependency not available in all CI
environments. It is stubbed at the sys.modules level before any app import.
"""

import io
import json
import os
import sys
import types
import unittest.mock as mock
from typing import cast

import boto3
import pytest
from fastapi import UploadFile
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")


# ---------------------------------------------------------------------------
# Stub `unstructured` before any app import touches it
# ---------------------------------------------------------------------------


def _install_unstructured_stub():
    if "unstructured" in sys.modules:
        return
    pkg = types.ModuleType("unstructured")
    partition_mod = types.ModuleType("unstructured.partition")
    auto_mod = types.ModuleType("unstructured.partition.auto")
    fake_element = types.SimpleNamespace(text="extracted text")
    setattr(auto_mod, "partition", lambda **kwargs: [fake_element])
    sys.modules["unstructured"] = pkg
    sys.modules["unstructured.partition"] = partition_mod
    sys.modules["unstructured.partition.auto"] = auto_mod


_install_unstructured_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeUpload:
    def __init__(self, filename="test.pdf", content=b"pdf bytes"):
        self.filename = filename
        self.file = io.BytesIO(content)
        self.content_type = "application/pdf"


def _create_sqs_queue():
    sqs = boto3.client("sqs", region_name="us-east-1")
    resp = sqs.create_queue(QueueName="test-queue")
    os.environ["QUEUE_URL"] = resp["QueueUrl"]
    return sqs, resp["QueueUrl"]


def _create_s3_bucket(name="rag-pipeline-upload-bucket"):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=name)
    return s3


def _reload_ingest(overrides=None):
    import importlib

    import backend.app.ingest as m

    importlib.reload(m)
    for k, v in (overrides or {}).items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# enqueue_file
# ---------------------------------------------------------------------------


class TestEnqueueFile:
    @mock_aws
    def test_message_lands_in_queue(self):
        sqs, queue_url = _create_sqs_queue()
        mod = _reload_ingest()
        mod.enqueue_file("uploads/report.pdf")
        messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        assert "Messages" in messages
        body = json.loads(messages["Messages"][0]["Body"])
        assert body["file"] == "uploads/report.pdf"

    @mock_aws
    def test_message_body_is_valid_json(self):
        sqs, queue_url = _create_sqs_queue()
        mod = _reload_ingest()
        mod.enqueue_file("any/key.pdf")
        messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        parsed = json.loads(messages["Messages"][0]["Body"])
        assert "file" in parsed

    @mock_aws
    def test_multiple_enqueues(self):
        sqs, queue_url = _create_sqs_queue()
        mod = _reload_ingest()
        mod.enqueue_file("a.pdf")
        mod.enqueue_file("b.pdf")
        messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
        assert len(messages.get("Messages", [])) == 2


# ---------------------------------------------------------------------------
# process_upload
# ---------------------------------------------------------------------------


def _fake_overrides():
    return {
        "index_document": mock.MagicMock(return_value=None),
        "get_embedding": lambda text: [0.1] * 1536,
        "OpenAIEmbeddings": lambda: mock.MagicMock(),
    }


class TestProcessUpload:
    @mock_aws
    def test_returns_message(self):
        _create_s3_bucket()
        mod = _reload_ingest(_fake_overrides())
        result = mod.process_upload(
            cast(UploadFile, _FakeUpload("report.pdf")), "user-123"
        )
        assert "message" in result

    @mock_aws
    def test_uploads_to_correct_s3_key(self):
        s3 = _create_s3_bucket()
        mod = _reload_ingest(_fake_overrides())
        mod.process_upload(
            cast(UploadFile, _FakeUpload("my.pdf", b"pdf-content")), "alice"
        )
        obj = s3.get_object(Bucket="rag-pipeline-upload-bucket", Key="alice/my.pdf")
        assert obj["Body"].read() == b"pdf-content"
