import json
import re
from io import BytesIO

from botocore.exceptions import ClientError

import boto3
from fastapi import HTTPException, UploadFile

from unstructured.partition.auto import partition

from .config import get_settings

settings = get_settings()
s3 = boto3.client("s3")

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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
            status_code=400,
            detail=f"Unsupported file extension: {extension}",
        )

    if (
        file.content_type
        and file.content_type not in settings.allowed_upload_mimes_list
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}",
        )

    content = file.file.read(settings.max_upload_size + 1)

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max size of {settings.max_upload_size} bytes",
        )

    text = content.decode(errors="ignore").lower()

    if any(pattern in text for pattern in settings.forbidden_upload_patterns_list):
        raise HTTPException(status_code=400, detail="Suspicious content detected")

    file.file.seek(0)
    return content


def upload_file_to_S3(file: UploadFile, user: str):
    content = validate_upload_file(file)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    safe_filename = sanitize_filename(file.filename)
    key = f"{user}/{safe_filename}"

    try:
        s3.upload_fileobj(BytesIO(content), settings.bucket_name, key)
    except ClientError:
        return key

    return key


def process_upload(file: UploadFile, user: str):
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    # read once
    content = validate_upload_file(file)

    # ✅ upload using fresh stream
    safe_filename = sanitize_filename(file.filename or "file")
    key = f"{user}/{safe_filename}"
    s3.upload_fileobj(BytesIO(content), settings.bucket_name, key)

    partition(file=BytesIO(content))

    # enqueue (safe)
    enqueue_file(key, user)

    return {"message": "queued"}


def enqueue_file(key: str, user: str = "test-user"):

    sqs = boto3.client("sqs")

    body = json.dumps({
        "bucket": settings.bucket_name,
        "key": key,
        "file": key,
        "user": user,
    })

    try:
        sqs.send_message(
            QueueUrl=settings.queue_url,
            MessageBody=body,
        )
    except ClientError:
        return body
