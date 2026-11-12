import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_TABLE_NAME = os.getenv("DYNAMODB_BM25_TABLE", "rag-bm25-cache")
_TTL_SECONDS = int(os.getenv("BM25_CACHE_TTL", str(24 * 3600)))  # 24 hours
_CORPUS_KEY = "bm25_corpus"
_MAX_RETRIES = 5

_dynamodb = None


def _table():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb").Table(_TABLE_NAME)
    return _dynamodb


# ── In-process warm cache (avoids DynamoDB hit on every Lambda invocation) ────
_warm_corpus: list[dict] | None = None
_warm_loaded_at: float = 0.0
_warm_version: int = 0
_WARM_TTL = 300  # re-fetch from DynamoDB every 5 minutes at most


def invalidate_warm_cache() -> None:
    """Force next get_corpus() to re-fetch from DynamoDB."""
    global _warm_corpus, _warm_loaded_at
    _warm_corpus = None
    _warm_loaded_at = 0.0


def _normalize_document(document: str | dict) -> dict:
    if isinstance(document, str):
        return {"text": document, "metadata": {}}
    if not isinstance(document, dict):
        raise ValueError("Invalid BM25 corpus entry")

    text = document.get("text", "")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    normalized = {"text": str(text), "metadata": metadata}
    if document.get("doc_id"):
        normalized["doc_id"] = str(document["doc_id"])
    return normalized


def get_corpus() -> list[dict]:
    """Return cached corpus. Falls back to empty list if cache miss."""
    global _warm_corpus, _warm_loaded_at, _warm_version

    now = time.time()
    if _warm_corpus is not None and (now - _warm_loaded_at) < _WARM_TTL:
        _warm_corpus = [_normalize_document(document) for document in _warm_corpus]
        return _warm_corpus

    try:
        resp = _table().get_item(Key={"pk": _CORPUS_KEY})
        item = resp.get("Item")
        if item:
            stored_corpus = json.loads(str(item.get("texts", "[]")))
            if not isinstance(stored_corpus, list):
                raise ValueError("BM25 corpus must be a JSON list")
            corpus = [_normalize_document(document) for document in stored_corpus]
            _warm_corpus = corpus
            _warm_loaded_at = now
            version_value = item.get("version", 0)
            if isinstance(version_value, (int, str)):
                _warm_version = int(version_value or 0)
            else:
                _warm_version = 0
            logger.debug(
                f"BM25 corpus loaded from DynamoDB: {len(corpus)} docs (v{_warm_version})"
            )
            return corpus
    except ClientError as e:
        logger.warning(f"BM25 corpus cache read failed: {e}")

    _warm_corpus = []
    _warm_loaded_at = now
    _warm_version = 0
    return []


def set_corpus(texts: list[str | dict]) -> None:
    """Persist corpus after ingest. Called by processor.py, not by search."""
    global _warm_corpus, _warm_loaded_at, _warm_version
    corpus = [_normalize_document(document) for document in texts]
    new_version = _warm_version + 1
    try:
        _table().put_item(
            Item={
                "pk": _CORPUS_KEY,
                "texts": json.dumps(corpus),
                "version": new_version,
                "ttl": int(time.time()) + _TTL_SECONDS,
            }
        )
        _warm_corpus = corpus
        _warm_loaded_at = time.time()
        _warm_version = new_version
        logger.info(f"BM25 corpus cached: {len(texts)} docs")
    except ClientError as e:
        logger.warning(f"BM25 corpus cache write failed: {e}")


def append_to_corpus(new_texts: list[str | dict]) -> None:
    """Add newly ingested chunks to the corpus with optimistic locking.

    Uses versioning to prevent data loss under concurrent writes.
    Retries automatically if version conflict detected.
    """
    global _warm_corpus, _warm_loaded_at, _warm_version

    if not new_texts:
        return

    normalized_new_texts = [_normalize_document(document) for document in new_texts]

    for attempt in range(_MAX_RETRIES):
        try:
            current = get_corpus()
            current_version = _warm_version

            updated = current + normalized_new_texts
            new_version = current_version + 1

            condition = "attribute_not_exists(#v) OR #v = :expected_version"

            _table().put_item(
                Item={
                    "pk": _CORPUS_KEY,
                    "texts": json.dumps(updated),
                    "version": new_version,
                    "ttl": int(time.time()) + _TTL_SECONDS,
                },
                ConditionExpression=condition,
                ExpressionAttributeNames={"#v": "version"},
                ExpressionAttributeValues={":expected_version": current_version},
            )

            _warm_corpus = updated
            _warm_loaded_at = time.time()
            _warm_version = new_version
            logger.debug(
                f"BM25 corpus appended: +{len(normalized_new_texts)} docs (v{current_version} → v{new_version})"
            )
            return

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "ConditionalCheckFailedException":
                _warm_corpus = None
                _warm_loaded_at = 0
                logger.debug(
                    f"BM25 append conflict (attempt {attempt + 1}/{_MAX_RETRIES}), retrying..."
                )
                time.sleep(0.1 * (2**attempt))
            else:
                logger.error(f"BM25 corpus cache write failed: {e}")
                return

    logger.error(f"BM25 append failed after {_MAX_RETRIES} retries (version conflict)")
