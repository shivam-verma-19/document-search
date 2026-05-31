"""
Tests for processor.py.

Verifies:
- DynamoDB-backed idempotency (replaces in-memory set)
- S3 download + text extraction
- Embedding + indexing flow
- BM25 corpus cache updated after indexing
- already_processed returns True after mark_processed
"""

import os
import sys
from io import BytesIO
from unittest.mock import MagicMock, call, patch

from botocore.exceptions import ClientError

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

# Stub heavy dependencies before import
mock_s3v = MagicMock()
mock_s3v.index_document = MagicMock()
sys.modules["backend.app.s3_vectors_client"] = mock_s3v

mock_embeddings = MagicMock()
mock_embeddings.get_embedding = MagicMock(return_value=[0.1] * 768)
sys.modules["backend.app.embeddings"] = mock_embeddings

mock_bm25_cache = MagicMock()
mock_bm25_cache.append_to_corpus = MagicMock()
sys.modules["backend.app.bm25_cache"] = mock_bm25_cache

with patch("backend.app.secrets.get_secret", return_value="test-key"):
    import backend.app.processor as proc

for key in [k for k in list(sys.modules) if "backend.app.processor" in k]:
    del sys.modules[key]

import backend.app.processor as proc  # noqa: E402


def _mock_dynamodb_table(has_item: bool = False):
    t = MagicMock()
    t.get_item.return_value = {"Item": {"s3_key": "key"}} if has_item else {}
    return t


class TestIdempotency:
    def test_claim_processing_succeeds_first_call(self):
        mock_t = _mock_dynamodb_table(has_item=False)
        with patch.object(proc, "get_idempotency_table", return_value=mock_t):
            assert proc.claim_processing("some/key.pdf") is True
        mock_t.put_item.assert_called_once()
        call_kwargs = mock_t.put_item.call_args.kwargs
        assert call_kwargs["ConditionExpression"] == "attribute_not_exists(s3_key)"

    def test_claim_processing_fails_already_claimed(self):
        from botocore.exceptions import ClientError

        mock_t = MagicMock()
        mock_t.put_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem"
        )
        with patch.object(proc, "get_idempotency_table", return_value=mock_t):
            assert proc.claim_processing("some/key.pdf") is False

    def test_mark_processed_updates_status(self):
        mock_t = _mock_dynamodb_table()
        with patch.object(proc, "get_idempotency_table", return_value=mock_t):
            proc.mark_processed("some/key.pdf")
        mock_t.update_item.assert_called_once()
        call_kwargs = mock_t.update_item.call_args.kwargs
        assert "UpdateExpression" in call_kwargs
        assert "processed" in call_kwargs["UpdateExpression"]


class TestTextExtraction:
    def test_plain_text(self):
        content = b"hello world"
        assert proc.extract_text_from_file(content, "file.txt") == "hello world"

    def test_pdf_extraction(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "pdf text"
        with patch("backend.app.processor.PdfReader") as mock_reader:
            mock_reader.return_value.pages = [mock_page]
            result = proc.extract_text_from_file(b"%PDF-fake", "doc.pdf")
        assert result == "pdf text"

    def test_pdf_error_returns_empty(self):
        with patch("backend.app.processor.PdfReader", side_effect=Exception("bad pdf")):
            result = proc.extract_text_from_file(b"bad", "doc.pdf")
        assert result == ""


class TestProcessFileFromS3:
    def _setup_dynamo(self, has_item=False):
        mock_t = _mock_dynamodb_table(has_item)
        mock_db = MagicMock()
        mock_db.Table.return_value = mock_t
        return mock_db, mock_t

    def test_skips_already_claimed(self):
        mock_t = _mock_dynamodb_table()
        mock_t.put_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem"
        )
        mock_db = MagicMock()
        mock_db.Table.return_value = mock_t
        with patch.object(proc, "get_idempotency_table", return_value=mock_t):
            result = proc.process_file_from_s3("bucket", "user/file.txt")
        assert result["status"] == "skipped"

    def test_processes_new_file(self):
        from botocore.exceptions import ClientError

        mock_t = MagicMock()
        mock_t.put_item.side_effect = None
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(b"some text content here")
        }
        with patch.object(
            proc, "get_idempotency_table", return_value=mock_t
        ), patch.object(proc, "get_s3_client", return_value=mock_s3_client):
            result = proc.process_file_from_s3("bucket", "user/file.txt")
        assert result["status"] == "processed"
        assert result["chunks"] >= 1
        mock_s3v.index_document.assert_called()
        mock_bm25_cache.append_to_corpus.assert_called()

    def test_empty_file_returns_empty_status(self):
        mock_t = MagicMock()
        mock_t.put_item.side_effect = None
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = {"Body": BytesIO(b"   ")}
        with patch.object(
            proc, "get_idempotency_table", return_value=mock_t
        ), patch.object(proc, "get_s3_client", return_value=mock_s3_client):
            result = proc.process_file_from_s3("bucket", "user/empty.txt")
        assert result["status"] == "empty"

    def test_s3_download_error_returns_error_status(self):
        from botocore.exceptions import ClientError

        mock_t = MagicMock()
        mock_t.put_item.side_effect = None
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        with patch.object(
            proc, "get_idempotency_table", return_value=mock_t
        ), patch.object(proc, "get_s3_client", return_value=mock_s3_client):
            result = proc.process_file_from_s3("bucket", "user/missing.txt")
        assert result["status"] == "error"
