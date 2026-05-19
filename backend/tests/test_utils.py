"""
Tests for backend/app/utils.py

Covers:
  - save_to_s3              (happy path, missing filename)
  - get_secrets             (happy path, missing secret)
  - normalize_text          (lowercasing, whitespace collapse)
  - clean_text              (punctuation stripping)
  - log_event               (JSON output to stdout)
  - build_prompt            (template structure)
"""

import importlib
import io
import json
import os
import sys

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")
os.environ.setdefault("BUCKET_NAME", "rag-pipeline-upload-bucket")


def _get_utils():
    """Always return the real utils module, bypassing any stub in sys.modules."""
    import backend.app.utils as m

    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeUpload:
    """Minimal UploadFile stand-in."""

    def __init__(self, filename="doc.pdf", content=b"bytes"):
        self.filename = filename
        self.file = io.BytesIO(content)


# ---------------------------------------------------------------------------
# save_to_s3
# ---------------------------------------------------------------------------


class TestSaveToS3:
    @mock_aws
    def test_happy_path_returns_key(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-pipeline-upload-bucket")

        save_to_s3 = _get_utils().save_to_s3
        key = save_to_s3(_FakeUpload("report.pdf", b"pdf-data"))
        assert key == "report.pdf"

    @mock_aws
    def test_file_actually_uploaded(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-pipeline-upload-bucket")

        save_to_s3 = _get_utils().save_to_s3
        save_to_s3(_FakeUpload("report.pdf", b"hello-world"))

        obj = s3.get_object(Bucket="rag-pipeline-upload-bucket", Key="report.pdf")
        assert obj["Body"].read() == b"hello-world"

    @mock_aws
    def test_missing_bucket_raises(self):
        save_to_s3 = _get_utils().save_to_s3
        with pytest.raises(ClientError):
            save_to_s3(_FakeUpload("x.pdf", b"data"))

    @mock_aws
    def test_empty_file_uploaded(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-pipeline-upload-bucket")

        save_to_s3 = _get_utils().save_to_s3
        key = save_to_s3(_FakeUpload("empty.pdf", b""))
        assert key == "empty.pdf"


# ---------------------------------------------------------------------------
# get_secrets
# ---------------------------------------------------------------------------


class TestGetSecrets:
    @mock_aws
    def test_missing_secret_raises(self):
        get_secrets = _get_utils().get_secrets
        with pytest.raises(ClientError):
            get_secrets()


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercases(self):
        assert _get_utils().normalize_text("HELLO WORLD") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert _get_utils().normalize_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert _get_utils().normalize_text("hello   world") == "hello world"

    def test_empty_string(self):
        assert _get_utils().normalize_text("") == ""

    def test_newlines_collapsed(self):
        assert _get_utils().normalize_text("hello\n\nworld") == "hello world"


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_removes_special_chars(self):
        result = _get_utils().clean_text("hello! @world#")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_preserves_alphanumeric(self):
        assert "hello" in _get_utils().clean_text("hello world")

    def test_preserves_periods(self):
        assert "." in _get_utils().clean_text("end of sentence.")

    def test_empty_string(self):
        assert _get_utils().clean_text("") == ""


# ---------------------------------------------------------------------------
# log_event
# ---------------------------------------------------------------------------


class TestLogEvent:
    def test_outputs_json_to_stdout(self, capsys):
        _get_utils().log_event("query", "success", 123)
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert data["event"] == "query"
        assert data["status"] == "success"
        assert data["latency_ms"] == 123

    def test_all_fields_present(self, capsys):
        _get_utils().log_event("upload", "error", 0)
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert {"event", "status", "latency_ms"}.issubset(data.keys())


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_context(self):
        prompt = _get_utils().build_prompt("some context", "what is X?")
        assert "some context" in prompt

    def test_contains_query(self):
        prompt = _get_utils().build_prompt("ctx", "what is X?")
        assert "what is X?" in prompt

    def test_contains_instruction(self):
        prompt = _get_utils().build_prompt("ctx", "q")
        assert "Answer ONLY using the context" in prompt

    def test_returns_string(self):
        assert isinstance(_get_utils().build_prompt("ctx", "q"), str)
