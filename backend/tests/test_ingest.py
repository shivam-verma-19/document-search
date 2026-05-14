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
from fastapi import HTTPException, UploadFile, status
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


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def _sanitize(self, name):
        from backend.app.ingest import sanitize_filename

        return sanitize_filename(name)

    def test_plain_name_unchanged(self):
        assert self._sanitize("report.pdf") == "report.pdf"

    def test_spaces_replaced_with_underscore(self):
        assert self._sanitize("my file.pdf") == "my_file.pdf"

    def test_special_chars_replaced(self):
        result = self._sanitize("file@name!.pdf")
        assert "@" not in result
        assert "!" not in result

    def test_leading_dots_stripped(self):
        result = self._sanitize("...secret.pdf")
        assert not result.startswith(".")

    def test_long_name_truncated_to_255(self):
        long_name = "a" * 300 + ".pdf"
        assert len(self._sanitize(long_name)) <= 255

    def test_whitespace_stripped(self):
        assert self._sanitize("  file.pdf  ") == "file.pdf"


# ---------------------------------------------------------------------------
# validate_upload_file – rejection branches
# ---------------------------------------------------------------------------


class TestValidateUploadFile:
    def _validate(self, upload):
        from backend.app.ingest import validate_upload_file

        return validate_upload_file(cast(UploadFile, upload))

    def test_missing_filename_raises_400(self):
        upload = _FakeUpload(filename="")
        with pytest.raises(HTTPException) as exc:
            self._validate(upload)
        assert exc.value.status_code == 400

    def test_no_extension_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            self._validate(_FakeUpload(filename="nodotfile"))
        assert exc.value.status_code == 400

    def test_disallowed_extension_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            self._validate(_FakeUpload(filename="malware.exe", content=b"data"))
        assert exc.value.status_code == 400

    def test_empty_file_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            self._validate(_FakeUpload(filename="empty.pdf", content=b""))
        assert exc.value.status_code == 400

    def test_file_too_large_raises_400(self):
        big = b"x" * (10 * 1024 * 1024 + 1)
        with pytest.raises(HTTPException) as exc:
            self._validate(_FakeUpload(filename="big.pdf", content=big))
        assert exc.value.status_code == 400

    def test_valid_pdf_returns_bytes(self):
        result = self._validate(_FakeUpload(filename="ok.pdf", content=b"data"))
        assert isinstance(result, bytes)
        assert result == b"data"


# ---------------------------------------------------------------------------
# upload_file_to_s3 – ClientError is swallowed
# ---------------------------------------------------------------------------


class TestUploadFileToS3ClientError:
    @mock_aws
    def test_client_error_is_swallowed_and_key_returned(self):
        """upload_file_to_s3 catches ClientError; bucket intentionally absent."""
        mod = _reload_ingest()
        # No bucket created → upload_fileobj raises ClientError
        result = mod.upload_file_to_s3(
            cast(UploadFile, _FakeUpload("report.pdf")), "user-1"
        )
        assert result == "user-1/report.pdf"
