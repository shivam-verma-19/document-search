from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

VECTOR_BUCKET_NAME = os.getenv("S3_VECTOR_BUCKET_NAME", "rag-vector-bucket")
VECTOR_INDEX_NAME = os.getenv("S3_VECTOR_INDEX_NAME", "rag-doc-index")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# Gemini gemini-embedding-001 outputs 768 dimensions.
# This MUST match the dimension configured in the S3 Vectors index (infra/s3.tf).
EXPECTED_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3vectors", region_name=AWS_REGION)  # type: ignore
    return _client


# ─── Public API ───────────────────────────────────────────────────────────────


def index_document(doc_id, text, embedding=None, metadata=None):
    """Store a single chunk with its embedding in S3 Vectors."""
    if embedding is None:
        raise ValueError("Embedding required")
    if metadata is None:
        metadata = {}

    if len(embedding) != EXPECTED_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, "
            f"expected {EXPECTED_DIMENSION}. "
            "Update EMBEDDING_DIMENSION env var and S3 Vectors index dimension to match."
        )

    _get_client().put_vectors(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        Vectors=[
            {
                "Key": str(doc_id),
                "Data": {"Float32": embedding},
                "Metadata": {"text": text, **metadata},
            }
        ],
    )
    return {"result": "created", "_id": str(doc_id)}


def search_documents(
    query_embedding,
    k=5,
    score_threshold: float = 0.0,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Return top-k documents as dicts with _id, _source, and similarity score.

    Args:
        query_embedding: Query vector.
        k: Number of results to fetch from S3 Vectors.
        score_threshold: Minimum similarity score to include a result (0.0 = no filter).
                         Typical cosine similarity: 0.0 (orthogonal) to 1.0 (identical).
                         A threshold of 0.5 filters out weakly related chunks.
        filters: Optional dict of metadata key-value filters (e.g., {"file_type": "pdf", "section": "intro"}).

    Returns:
        List of dicts: {_id, _source: {text, metadata}, score}
        Ordered best-first, filtered by score_threshold and metadata filters.
    """
    fetch_k = max(k * 4, 20) if filters else k
    response = _get_client().query_vectors(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        QueryVector={"Float32": query_embedding},
        TopK=fetch_k,
        ReturnMetadata=True,
    )
    docs = []
    for item in response.get("Vectors", []):
        meta = item.get("Metadata", {})
        # S3 Vectors returns Score in the query response
        score = float(item.get("Score", 1.0))

        if score < score_threshold:
            logger.debug(
                f"Filtered chunk {item.get('Key')} — score {score:.3f} < threshold {score_threshold:.3f}"
            )
            continue

        if filters:
            match = True
            for f_key, f_val in filters.items():
                if f_val is None:
                    continue
                if f_key == "date_from":
                    up_date = str(meta.get("upload_date", ""))
                    if up_date and up_date < str(f_val):
                        match = False
                        break
                elif f_key == "date_to":
                    up_date = str(meta.get("upload_date", ""))
                    if up_date and up_date > str(f_val):
                        match = False
                        break
                elif f_key in ("section", "section_header"):
                    sec = str(meta.get("section_header", "")).lower()
                    if str(f_val).lower() not in sec:
                        match = False
                        break
                else:
                    doc_val = str(meta.get(f_key, "")).lower()
                    if doc_val != str(f_val).lower():
                        match = False
                        break
            if not match:
                continue

        docs.append(
            {
                "_id": item["Key"],
                "_source": {
                    "text": meta.get("text", ""),
                    "metadata": {
                        k_meta: v for k_meta, v in meta.items() if k_meta != "text"
                    },
                },
                "score": score,
            }
        )
        if len(docs) >= k:
            break
    return docs


def search_similar(query_embedding, k=5) -> list[str]:
    """Return top-k chunk texts (for backward compatibility)."""
    docs = search_documents(query_embedding, k)
    return [d["_source"]["text"] for d in docs]


def get_document(doc_id):
    """Fetch a single document by key."""
    response = _get_client().get_vectors(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        Keys=[str(doc_id)],
    )
    items = response.get("Vectors", [])
    if not items:
        return None
    item = items[0]
    meta = item.get("Metadata", {})
    return {
        "_id": item["Key"],
        "_source": {
            "text": meta.get("text", ""),
            "metadata": {k: v for k, v in meta.items() if k != "text"},
        },
    }


def get_documents_by_doc_base_id(doc_base_id: str) -> list[dict]:
    """Return all chunks that belong to a logical document."""
    paginator = _get_client().get_paginator("list_vectors")
    prefix = f"{doc_base_id}#"
    docs: list[dict] = []
    for page in paginator.paginate(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        ReturnMetadata=True,
    ):
        for vector in page.get("Vectors", []):
            key = str(vector.get("Key", ""))
            if key.startswith(prefix):
                meta = vector.get("Metadata", {})
                docs.append(
                    {
                        "_id": key,
                        "_source": {
                            "text": meta.get("text", ""),
                            "metadata": {k: v for k, v in meta.items() if k != "text"},
                        },
                    }
                )
    return docs


def _find_keys_for_doc_base_id(
    doc_base_id: str, user_id: str | None = None
) -> list[str]:
    paginator = _get_client().get_paginator("list_vectors")
    prefix = f"{doc_base_id}#"
    keys: list[str] = []
    for page in paginator.paginate(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        ReturnMetadata=user_id is not None,
    ):
        for vector in page.get("Vectors", []):
            key = str(vector.get("Key", ""))
            owner_id = vector.get("Metadata", {}).get("user_id")
            if key.startswith(prefix) and (user_id is None or owner_id == user_id):
                keys.append(key)
    return keys


def delete_document(doc_id, user_id: str | None = None):
    """Remove a document or document chunk from the index."""
    key_str = str(doc_id)
    if "#" in key_str:
        if user_id is None:
            keys = [key_str]
        else:
            document = get_document(key_str)
            metadata = document.get("_source", {}).get("metadata", {}) if document else {}
            keys = [key_str] if metadata.get("user_id") == user_id else []
    else:
        keys = _find_keys_for_doc_base_id(key_str, user_id=user_id)

    if not keys:
        return {"result": "not_found", "_id": key_str, "deleted_count": 0}

    _get_client().delete_vectors(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        Keys=keys,
    )
    return {"result": "deleted", "_id": key_str, "deleted_count": len(keys)}


def reset_collection():
    """Delete all vectors — used in tests / re-index flows."""
    paginator = _get_client().get_paginator("list_vectors")
    keys = []
    for page in paginator.paginate(
        VectorBucketName=VECTOR_BUCKET_NAME, IndexName=VECTOR_INDEX_NAME
    ):
        keys.extend(v["Key"] for v in page.get("Vectors", []))
    if keys:
        _get_client().delete_vectors(
            VectorBucketName=VECTOR_BUCKET_NAME,
            IndexName=VECTOR_INDEX_NAME,
            Keys=keys,
        )


def generate_doc_id():
    return str(uuid.uuid4())


def get_all_documents() -> list[str]:
    """Return all stored chunk texts for BM25 corpus building."""
    paginator = _get_client().get_paginator("list_vectors")
    texts = []
    for page in paginator.paginate(
        VectorBucketName=VECTOR_BUCKET_NAME,
        IndexName=VECTOR_INDEX_NAME,
        ReturnMetadata=True,
    ):
        for v in page.get("Vectors", []):
            t = v.get("Metadata", {}).get("text", "")
            if t:
                texts.append(t)
    return texts
