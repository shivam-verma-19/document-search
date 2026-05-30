import io
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from backend.app.ingest import (
    upload_file_to_s3,
    validate_upload_file,
)


class TestValidateUploadFile:
    def test_valid_file_returns_content(self):
        upload = UploadFile(
            filename="report.pdf",
            file=BytesIO(b"hello world"),
            headers=Headers({"content-type": "application/pdf"}),
        )

        content = validate_upload_file(upload)

        assert content == b"hello world"


class TestUploadFileToS3:
    @patch("backend.app.ingest.get_s3_client")
    def test_returns_s3_key(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        upload = UploadFile(
            filename="report.pdf",
            file=BytesIO(b"hello world"),
            headers=Headers({"content-type": "application/pdf"}),
        )

        key = upload_file_to_s3(upload, "user-123")

        assert key == "user-123/report.pdf"

    @patch("backend.app.ingest.get_s3_client")
    def test_calls_upload_fileobj(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        upload = UploadFile(
            filename="report.pdf",
            file=BytesIO(b"hello world"),
            headers=Headers({"content-type": "application/pdf"}),
        )

        upload_file_to_s3(upload, "user-123")

        assert mock_client.upload_fileobj.called


class TestUploadFileToS3ClientError:
    @patch("backend.app.ingest.get_s3_client")
    def test_client_error_is_swallowed_and_key_returned(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client.upload_fileobj.side_effect = Exception("upload failed")
        mock_client_factory.return_value = mock_client

        upload = UploadFile(
            filename="report.pdf",
            file=BytesIO(b"hello world"),
            headers=Headers({"content-type": "application/pdf"}),
        )

        with pytest.raises(Exception):
            upload_file_to_s3(upload, "user-123")


class TestIngest:
    def _make_upload(
        self,
        filename="test.pdf",
        content=b"file content",
        content_type="application/pdf",
    ):
        f = MagicMock()
        f.filename = filename
        f.content_type = content_type
        f.file = io.BytesIO(content)
        return f

    def test_sanitize_filename_strips_bad_chars(self):
        from backend.app.ingest import sanitize_filename

        assert sanitize_filename("hello world!.pdf") == "hello_world_.pdf"
        assert sanitize_filename("  file  ") == "file"

    def test_sanitize_filename_max_length(self):
        from backend.app.ingest import sanitize_filename

        long = "a" * 300 + ".pdf"
        result = sanitize_filename(long)
        assert len(result) <= 255

    def test_validate_upload_missing_filename(self):
        from fastapi import HTTPException

        from backend.app.ingest import validate_upload_file

        f = MagicMock()
        f.filename = ""
        with pytest.raises(HTTPException) as exc:
            validate_upload_file(f)
        assert exc.value.status_code == 400

    def test_validate_upload_no_extension(self):
        from fastapi import HTTPException

        from backend.app.ingest import validate_upload_file

        f = MagicMock()
        f.filename = "noextension"
        f.content_type = "text/plain"
        with pytest.raises(HTTPException) as exc:
            validate_upload_file(f)
        assert exc.value.status_code == 400

    def test_validate_upload_unsupported_extension(self):
        from fastapi import HTTPException

        from backend.app.ingest import validate_upload_file

        f = self._make_upload(
            filename="virus.exe", content_type="application/octet-stream"
        )
        with pytest.raises(HTTPException) as exc:
            validate_upload_file(f)
        assert exc.value.status_code == 400

    def test_validate_upload_empty_file(self):
        from fastapi import HTTPException

        from backend.app.ingest import validate_upload_file

        f = self._make_upload(content=b"")
        with pytest.raises(HTTPException) as exc:
            validate_upload_file(f)
        assert exc.value.status_code == 400

    def test_validate_upload_file_too_large(self, monkeypatch):
        from fastapi import HTTPException

        from backend.app.config import get_settings
        from backend.app.ingest import validate_upload_file

        settings = get_settings()
        big = b"x" * (settings.max_upload_size + 2)
        f = self._make_upload(content=big)
        with pytest.raises(HTTPException) as exc:
            validate_upload_file(f)
        assert exc.value.status_code == 400

    def test_validate_upload_valid_file_returns_bytes(self):
        from backend.app.ingest import validate_upload_file

        f = self._make_upload(
            content=b"hello world", filename="doc.txt", content_type="text/plain"
        )
        result = validate_upload_file(f)
        assert isinstance(result, bytes)
        assert b"hello" in result

    def test_upload_file_to_s3_success(self):
        from backend.app.ingest import upload_file_to_s3

        f = self._make_upload(
            content=b"data", filename="doc.txt", content_type="text/plain"
        )
        with patch("backend.app.ingest.get_s3_client") as mock_s3:
            key = upload_file_to_s3(f, "user123")
        assert "user123" in key
        assert "doc" in key

    def test_upload_file_to_s3_client_error(self):
        from botocore.exceptions import ClientError
        from fastapi import HTTPException

        from backend.app.ingest import upload_file_to_s3

        f = self._make_upload(
            content=b"data", filename="doc.txt", content_type="text/plain"
        )
        with patch("backend.app.ingest.get_s3_client") as mock_s3:
            mock_s3.return_value.upload_fileobj.side_effect = ClientError(
                {"Error": {"Code": "NoSuchBucket", "Message": "err"}}, "PutObject"
            )
            with pytest.raises(HTTPException) as exc:
                upload_file_to_s3(f, "user123")
        assert exc.value.status_code == 500

    def test_enqueue_file_success(self):
        from backend.app.ingest import enqueue_file

        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.send_message.return_value = {}
            enqueue_file("user/file.txt", "user123")  # should not raise

    def test_enqueue_file_sqs_error_is_non_fatal(self):
        from botocore.exceptions import ClientError

        from backend.app.ingest import enqueue_file

        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.send_message.side_effect = ClientError(
                {"Error": {"Code": "QueueDoesNotExist", "Message": "err"}},
                "SendMessage",
            )
            enqueue_file("user/file.txt")  # should NOT raise
