import json
import logging
import re
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, UploadFile

from .config import get_settings
from .utils import clean_text

logger = logging.getLogger(__name__)
settings = get_settings()

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def get_s3_client():
    return boto3.client("s3")  # type: ignore


def sanitize_filename(filename: str) -> str:
    filename = filename.strip()
    filename = _FILENAME_SANITIZE_RE.sub("_", filename)
    return filename[:255].strip("._")


def validate_upload_file(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if "." not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid file extension")

    extension = file.filename.rsplit(".", 1)[-1].lower()
    if extension not in settings.allowed_upload_extensions_list:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file extension: {extension}"
        )

    if (
        file.content_type
        and file.content_type not in settings.allowed_upload_mimes_list
    ):
        raise HTTPException(
            status_code=400, detail=f"Unsupported content type: {file.content_type}"
        )

    # Read raw bytes
    raw = file.file.read(settings.max_upload_size + 1)

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # FIX: clean content BEFORE the size check so both limits apply to
    # the same bytes that will actually be stored.
    try:
        decoded = raw.decode("utf-8", errors="ignore")
        decoded = clean_text(decoded)
        content = decoded.encode("utf-8")
    except Exception as e:
        logger.warning(f"Text cleaning failed, proceeding with original: {e}")
        content = raw

    if len(content) > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="File too large")

    return content


def upload_file_to_s3(file: UploadFile, user: str) -> str:
    """Store file in S3 and return the object key."""
    content = validate_upload_file(file)
    safe_filename = sanitize_filename(file.filename or "file")
    key = f"{user}/{safe_filename}"
    try:
        get_s3_client().upload_fileobj(BytesIO(content), settings.bucket_name, key)
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")
    return key


def enqueue_file(key: str, user: str = "anonymous") -> None:
    """Push an SQS message so the worker Lambda picks up and indexes the file."""
    sqs = boto3.client("sqs")  # type: ignore
    body = json.dumps({"bucket": settings.bucket_name, "key": key, "user": user})
    try:
        sqs.send_message(QueueUrl=settings.queue_url, MessageBody=body)
        logger.info(f"Enqueued {key} for processing")
    except ClientError as e:
        logger.error(f"SQS enqueue failed for {key}: {e}")
        # Non-fatal: file is in S3, can be re-queued manually
