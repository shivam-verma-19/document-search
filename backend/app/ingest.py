import json
import re
from io import BytesIO
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, UploadFile

from .chunker import chunk_text
from .config import get_settings
from .embeddings import get_embedding
from .faiss_client import index_document

settings = get_settings()
s3 = boto3.client("s3")  # type: ignore

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
            status_code=400, detail=f"Unsupported file extension: {extension}"
        )

    if (
        file.content_type
        and file.content_type not in settings.allowed_upload_mimes_list
    ):
        raise HTTPException(
            status_code=400, detail=f"Unsupported content type: {file.content_type}"
        )

    content = file.file.read(settings.max_upload_size + 1)

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="File too large")

    file.file.seek(0)
    return content


def upload_file_to_s3(file: UploadFile, user: str):
    content = validate_upload_file(file)

    safe_filename = sanitize_filename(file.filename or "file")
    key = f"{user}/{safe_filename}"

    try:
        s3.upload_fileobj(BytesIO(content), settings.bucket_name, key)
    except ClientError:
        pass

    return key


def process_upload(file: UploadFile, user: str):
    content = validate_upload_file(file)

    safe_filename = sanitize_filename(file.filename or "file")
    key = f"{user}/{safe_filename}"

    # Upload
    s3.upload_fileobj(BytesIO(content), settings.bucket_name, key)

    # Extract text
    text = content.decode(errors="ignore")

    doc_id = str(uuid4())
    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        index_document(
            doc_id=doc_id,
            text=chunk,
            embedding=embedding,
            metadata={
                "user_id": key.split("/")[0],
                "chunk_id": idx,
            },
        )

    enqueue_file(key, user)

    return {"message": "queued"}


def enqueue_file(key: str, user: str = "test-user"):
    sqs = boto3.client("sqs")

    body = json.dumps(
        {
            "bucket": settings.bucket_name,
            "key": key,
            "file": key,
            "user": user,
        }
    )

    try:
        sqs.send_message(
            QueueUrl=settings.queue_url,
            MessageBody=body,
        )
    except ClientError:
        return None

    return body
