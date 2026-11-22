import json
import logging
import re
import uuid
from io import BytesIO

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException, UploadFile

from .config import get_settings
from .utils import clean_text

logger = logging.getLogger(__name__)
settings = get_settings()

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MIME_TYPES_BY_EXTENSION = {
    "pdf": {"application/pdf"},
    "txt": {"text/plain"},
    "csv": {"text/csv", "application/vnd.ms-excel"},
}


def get_s3_client():
    return boto3.client("s3")  # type: ignore


def sanitize_filename(filename: str) -> str:
    filename = filename.strip()
    filename = _FILENAME_SANITIZE_RE.sub("_", filename)
    return filename[:255].strip("._")


def validate_upload_details(filename: str, content_type: str, size: int) -> None:
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if size > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="File too large")

    extension = filename.rsplit(".", 1)[-1].lower()
    if (
        extension not in settings.allowed_upload_extensions_list
        or extension not in _MIME_TYPES_BY_EXTENSION
    ):
        raise HTTPException(status_code=415, detail="Unsupported file extension")
    if (
        content_type not in settings.allowed_upload_mimes_list
        or content_type not in _MIME_TYPES_BY_EXTENSION[extension]
    ):
        raise HTTPException(status_code=415, detail="Unsupported file content type.")


def create_presigned_upload(user: str, filename: str, content_type: str, size: int) -> dict:
    validate_upload_details(filename, content_type, size)
    safe_filename = sanitize_filename(filename)
    key = f"{user}/{uuid.uuid4().hex}_{safe_filename}"
    try:
        presigned_post = get_s3_client().generate_presigned_post(
            Bucket=settings.bucket_name,
            Key=key,
            Fields={"Content-Type": content_type},
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, size],
            ],
            ExpiresIn=300,
        )
    except (BotoCoreError, ClientError) as e:
        logger.error("Could not create presigned upload URL", exc_info=True)
        raise HTTPException(
            status_code=503, detail="File upload is temporarily unavailable."
        ) from e
    return {
        "upload_url": presigned_post["url"],
        "fields": presigned_post["fields"],
        "key": key,
        "content_type": content_type,
    }


def validate_uploaded_object(key: str, user: str) -> None:
    if not key.startswith(f"{user}/"):
        raise HTTPException(status_code=404, detail="Uploaded file not found.")

    try:
        metadata = get_s3_client().head_object(Bucket=settings.bucket_name, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            raise HTTPException(
                status_code=404, detail="Uploaded file not found."
            ) from e
        logger.error("Could not verify uploaded object", exc_info=True)
        raise HTTPException(
            status_code=503, detail="File upload is temporarily unavailable."
        ) from e
    except BotoCoreError as e:
        logger.error("Could not verify uploaded object", exc_info=True)
        raise HTTPException(
            status_code=503, detail="File upload is temporarily unavailable."
        ) from e

    filename = key.rsplit("/", 1)[-1].split("_", 1)[-1]
    validate_upload_details(
        filename,
        metadata.get("ContentType", ""),
        int(metadata.get("ContentLength", 0)),
    )


def validate_upload_file(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if "." not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid file extension")

    extension = file.filename.rsplit(".", 1)[-1].lower()
    if extension not in settings.allowed_upload_extensions_list:
        raise HTTPException(
            status_code=415, detail=f"Unsupported file extension: {extension}"
        )

    if (
        file.content_type
        and file.content_type not in settings.allowed_upload_mimes_list
    ):
        raise HTTPException(status_code=415, detail="Unsupported file content type.")

    # Read one byte beyond the limit so oversized files are rejected, not truncated.
    raw = file.file.read(settings.max_upload_size + 1)

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="File too large")

    if extension == "pdf":
        if not raw.startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail="Invalid PDF file.")
        return raw

    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=415, detail="Text files must use UTF-8 encoding."
        ) from exc

    return clean_text(decoded).encode("utf-8")


def upload_file_to_s3(file: UploadFile, user: str) -> str:
    """Store file in S3 and return the object key."""
    content = validate_upload_file(file)
    safe_filename = sanitize_filename(file.filename or "file")
    key = f"{user}/{uuid.uuid4().hex}_{safe_filename}"
    try:
        get_s3_client().upload_fileobj(BytesIO(content), settings.bucket_name, key)
    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(
            status_code=503, detail="File upload is temporarily unavailable."
        ) from e
    return key


def enqueue_file(key: str, user: str = "anonymous") -> None:
    """Push an SQS message so the worker Lambda picks up and indexes the file."""
    sqs = boto3.client("sqs")  # type: ignore
    body = json.dumps({"bucket": settings.bucket_name, "key": key, "user": user})
    try:
        sqs.send_message(QueueUrl=settings.queue_url, MessageBody=body)
        logger.info(f"Enqueued {key} for processing")
    except (BotoCoreError, ClientError) as e:
        logger.error(f"SQS enqueue failed for {key}: {e}")
        raise HTTPException(
            status_code=503, detail="File processing could not be queued."
        ) from e
