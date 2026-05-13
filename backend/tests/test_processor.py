"""
Tests for backend/app/processor.py and backend/app/worker_lambda.py

processor covers:
  - already_processed / mark_processed idempotency
  - process_file_from_s3: skip when already processed
  - process_file_from_s3: returns "empty" when extracted text is blank
  - process_file_from_s3: happy path indexes chunks and marks key processed

worker_lambda covers:
  - handler with empty Records list
  - handler processes each record and passes bucket + key
  - handler tolerates missing Records key
"""

import importlib
import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("OPENSEARCH_ENDPOINT", "http://mock-opensearch")
os.environ.setdefault("BUCKET_NAME", "rag-pipeline-upload-bucket")


# ---------------------------------------------------------------------------
# Stub heavy deps before importing processor
# ---------------------------------------------------------------------------


def _stub_if_missing(name, obj):
    if name not in sys.modules:
        sys.modules[name] = obj


_stub_unstructured_pkg = types.ModuleType("unstructured")
_stub_partition_mod = types.ModuleType("unstructured.partition")
_stub_auto_mod = types.ModuleType("unstructured.partition.auto")
_stub_auto_mod.partition = lambda **kwargs: [
    types.SimpleNamespace(text="extracted text")
]
_stub_if_missing("unstructured", _stub_unstructured_pkg)
_stub_if_missing("unstructured.partition", _stub_partition_mod)
_stub_if_missing("unstructured.partition.auto", _stub_auto_mod)

_stub_if_missing("langchain_openai", MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_processor(overrides=None):
    import backend.app.processor as proc

    importlib.reload(proc)
    proc.processed_files.clear()
    for k, v in (overrides or {}).items():
        setattr(proc, k, v)
    return proc


def _reload_worker(overrides=None):
    import backend.app.worker_lambda as wl

    importlib.reload(wl)
    for k, v in (overrides or {}).items():
        setattr(wl, k, v)
    return wl


# ===========================================================================
# processor – already_processed / mark_processed
# ===========================================================================


class TestProcessorIdempotency:
    def test_already_processed_false_initially(self):
        proc = _reload_processor()
        assert proc.already_processed("some/key.pdf") is False

    def test_mark_processed_makes_key_recognized(self):
        proc = _reload_processor()
        proc.mark_processed("some/key.pdf")
        assert proc.already_processed("some/key.pdf") is True

    def test_different_keys_are_independent(self):
        proc = _reload_processor()
        proc.mark_processed("a.pdf")
        assert proc.already_processed("b.pdf") is False

    def test_marking_twice_is_safe(self):
        proc = _reload_processor()
        proc.mark_processed("file.pdf")
        proc.mark_processed("file.pdf")  # should not raise
        assert proc.already_processed("file.pdf") is True


# ===========================================================================
# processor – process_file_from_s3
# ===========================================================================


class TestProcessFileFromS3:
    @mock_aws
    def test_returns_skipped_when_already_processed(self):
        proc = _reload_processor()
        proc.mark_processed("uploads/doc.pdf")
        result = proc.process_file_from_s3("my-bucket", "uploads/doc.pdf")
        assert result == {"status": "skipped", "key": "uploads/doc.pdf"}

    @mock_aws
    def test_returns_empty_when_no_text_extracted(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="my-bucket")
        s3.put_object(Bucket="my-bucket", Key="uploads/blank.txt", Body=b"   ")

        proc = _reload_processor()
        blank_el = types.SimpleNamespace(text="")

        with patch("backend.app.processor.partition", return_value=[blank_el]):
            result = proc.process_file_from_s3("my-bucket", "uploads/blank.txt")

        assert result["status"] == "empty"

    @mock_aws
    def test_happy_path_indexes_chunks(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="my-bucket")
        s3.put_object(Bucket="my-bucket", Key="user1/doc.txt", Body=b"hello world text")

        index_calls = []
        proc = _reload_processor(
            {
                "get_embedding": lambda t: [0.1] * 3,
                "index_document": lambda **kw: index_calls.append(kw),
            }
        )

        el = types.SimpleNamespace(text="hello world text")
        with patch("backend.app.processor.partition", return_value=[el]):
            result = proc.process_file_from_s3("my-bucket", "user1/doc.txt")

        assert result["status"] == "processed"
        assert result["chunks"] >= 1
        assert len(index_calls) >= 1

    @mock_aws
    def test_happy_path_marks_key_processed(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="my-bucket")
        s3.put_object(Bucket="my-bucket", Key="user1/doc.txt", Body=b"some content")

        proc = _reload_processor(
            {
                "get_embedding": lambda t: [0.1] * 3,
                "index_document": lambda **kw: None,
            }
        )

        el = types.SimpleNamespace(text="some content")
        with patch("backend.app.processor.partition", return_value=[el]):
            proc.process_file_from_s3("my-bucket", "user1/doc.txt")

        assert proc.already_processed("user1/doc.txt")

    @mock_aws
    def test_second_call_with_same_key_is_skipped(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="my-bucket")
        s3.put_object(Bucket="my-bucket", Key="user1/doc.txt", Body=b"content")

        index_calls = []
        proc = _reload_processor(
            {
                "get_embedding": lambda t: [],
                "index_document": lambda **kw: index_calls.append(kw),
            }
        )

        el = types.SimpleNamespace(text="content")
        with patch("backend.app.processor.partition", return_value=[el]):
            proc.process_file_from_s3("my-bucket", "user1/doc.txt")
            calls_after_first = len(index_calls)
            result = proc.process_file_from_s3("my-bucket", "user1/doc.txt")

        assert result["status"] == "skipped"
        assert len(index_calls) == calls_after_first  # no extra indexing


# ===========================================================================
# worker_lambda – handler
# ===========================================================================


class TestWorkerLambdaHandler:
    def test_empty_records_returns_zero_processed(self):
        wl = _reload_worker()
        result = wl.handler({"Records": []}, None)
        assert result == {"processed": 0}

    def test_missing_records_key_returns_zero_processed(self):
        wl = _reload_worker()
        result = wl.handler({}, None)
        assert result == {"processed": 0}

    def test_processes_each_record(self):
        calls = []

        def fake_process(bucket, key):
            calls.append((bucket, key))
            return {"status": "processed"}

        wl = _reload_worker({"process_file_from_s3": fake_process})
        event = {
            "Records": [
                {"body": json.dumps({"bucket": "b1", "key": "k1"})},
                {"body": json.dumps({"bucket": "b2", "key": "k2"})},
            ]
        }
        result = wl.handler(event, None)

        assert result == {"processed": 2}
        assert ("b1", "k1") in calls
        assert ("b2", "k2") in calls

    def test_processed_count_matches_record_count(self):
        wl = _reload_worker(
            {"process_file_from_s3": lambda b, k: {"status": "processed"}}
        )
        records = [
            {"body": json.dumps({"bucket": "b", "key": f"k{i}"})} for i in range(5)
        ]
        result = wl.handler({"Records": records}, None)
        assert result["processed"] == 5
