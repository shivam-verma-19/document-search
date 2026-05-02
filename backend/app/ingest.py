import json
import re
from io import BytesIO

import boto3
from fastapi import HTTPException, UploadFile

from .config import get_settings

settings = get_settings()
s3 = boto3.client("s3")
sqs = boto3.client("sqs")

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
    if extension not in settings.allowed_upload_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {extension}",
        )

    if file.content_type not in settings.allowed_upload_mimes:
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

    lower = content.lower()
    if any(pattern in lower for pattern in settings.forbidden_upload_patterns):
        raise HTTPException(status_code=400, detail="Suspicious content detected")

    file.file.seek(0)
    return content


def upload_file_to_s3(file: UploadFile, user: str):
    content = validate_upload_file(file)
    safe_filename = sanitize_filename(file.filename)
    key = f"{user}/{safe_filename}"

    s3.upload_fileobj(BytesIO(content), settings.bucket_name, key)
    return key


def enqueue_file(bucket: str, key: str, user: str):
    sqs.send_message(
        QueueUrl=settings.queue_url,
        MessageBody=json.dumps({"bucket": bucket, "key": key, "user": user}),
    )