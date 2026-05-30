import logging
import os
import time
import uuid
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from pypdf import PdfReader

from .bm25_cache import append_to_corpus
from .chunker import chunk_text
from .config import get_settings
from .embeddings import get_embedding
from .s3_vectors_client import index_document

logger = logging.getLogger(__name__)
settings = get_settings()

_s3_client = None
_dynamodb_resource = None
_IDEMPOTENCY_TABLE = os.getenv("DYNAMODB_IDEMPOTENCY_TABLE", "rag-processed-keys")


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def get_dynamodb_resource():
    global _dynamodb_resource
    if _dynamodb_resource is None:
        _dynamodb_resource = boto3.resource("dynamodb")
    return _dynamodb_resource


def get_idempotency_table():
    return get_dynamodb_resource().Table(_IDEMPOTENCY_TABLE)


# ─── DynamoDB-backed idempotency ──────────────────────────────────────────────


def already_processed(key: str) -> bool:
    try:
        table = get_idempotency_table()
        resp = table.get_item(Key={"s3_key": key})
        return "Item" in resp
    except ClientError as e:
        logger.warning(f"Idempotency check failed, assuming not processed: {e}")
        return False


def claim_processing(key: str) -> bool:
    try:
        table = get_idempotency_table()
        table.put_item(
            Item={
                "s3_key": key,
                "status": "processing",
                "processing_started": int(time.time()),
                "ttl": int(time.time()) + 7 * 24 * 3600,
            },
            ConditionExpression="attribute_not_exists(s3_key)",
        )
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "ConditionalCheckFailedException":
            return False
        # FIX: log and return False instead of re-raising, so the Lambda
        # returns a structured error result rather than crashing into DLQ.
        logger.warning(f"Failed to claim idempotency row for {key}: {e}")
        return False


def mark_processed(key: str) -> None:
    try:
        table = get_idempotency_table()
        table.update_item(
            Key={"s3_key": key},
            UpdateExpression="SET #status = :processed, processed_at = :processed_at, ttl = :ttl",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":processed": "processed",
                ":processed_at": int(time.time()),
                ":ttl": int(time.time()) + 7 * 24 * 3600,
            },
        )
    except ClientError as e:
        logger.warning(f"Failed to mark key as processed: {e}")


# ─── Text extraction ──────────────────────────────────────────────────────────


def extract_text_from_file(content: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return _extract_from_pdf(content)
    return content.decode("utf-8", errors="ignore")


def _extract_from_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
        return ""


# ─── Pipeline steps ───────────────────────────────────────────────────────────


def _download_from_s3(bucket: str, key: str) -> bytes:
    obj = get_s3_client().get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _chunk_and_embed(text: str, key: str) -> tuple[list[Document], str, list[str]]:
    """Return (chunks, doc_base_id, chunk_texts)."""
    chunks = [Document(page_content=c) for c in chunk_text(text)]
    doc_base_id = str(uuid.uuid4())
    new_texts: list[str] = []

    for idx, chunk in enumerate(chunks):
        chunk_doc_id = f"{doc_base_id}#{idx}"
        embedding = get_embedding(chunk.page_content)
        index_document(
            doc_id=chunk_doc_id,
            text=chunk.page_content,
            embedding=embedding,
            metadata={
                "user_id": key.split("/")[0],
                "chunk_id": idx,
                "doc_base_id": doc_base_id,
            },
        )
        new_texts.append(chunk.page_content)

    return chunks, doc_base_id, new_texts


# ─── Main processor (called by SQS worker Lambda) ────────────────────────────


def process_file_from_s3(bucket: str, key: str) -> dict:
    if not claim_processing(key):
        logger.info(f"Skipping already-claimed key: {key}")
        return {"status": "skipped", "key": key}

    try:
        content = _download_from_s3(bucket, key)
    except ClientError as e:
        logger.error(f"S3 download failed for {key}: {e}")
        return {"status": "error", "key": key, "reason": str(e)}

    text = extract_text_from_file(content, key)
    if not text.strip():
        logger.warning(f"No extractable text in {key}")
        return {"status": "empty", "key": key}

    chunks, doc_base_id, new_texts = _chunk_and_embed(text, key)

    append_to_corpus(new_texts)

    mark_processed(key)
    logger.info(f"Processed {key}: {len(chunks)} chunks, doc_base_id={doc_base_id}")
    return {
        "status": "processed",
        "chunks": len(chunks),
        "key": key,
        "doc_id": doc_base_id,
    }
