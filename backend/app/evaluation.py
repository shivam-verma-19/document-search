"""
Legacy evaluation module maintaining backward compatibility for store_eval
while integrating with the RAGAS evaluation harness in eval.py.
"""

from __future__ import annotations

import time
import uuid

import boto3

from .eval import (
    AnswerMetrics,
    EvalRecord,
    RetrievalMetrics,
    evaluate_answer,
    evaluate_retrieval,
    get_eval_records,
    log_eval,
)

_dynamodb = None
_table = None


def _get_table():
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb")
        _table = _dynamodb.Table("rag-eval")
    return _table


def store_eval(query: str, latency: int | float, precision: int | float) -> None:
    """Legacy helper writing latency and precision to rag-eval table."""
    now = int(time.time())
    record_id = str(uuid.uuid4())
    _get_table().put_item(
        Item={
            "query": query,
            "eval_id": record_id,
            "id": record_id,
            "timestamp": now,
            "ttl": now + 90 * 24 * 3600,
            "latency": latency,
            "precision": precision,
        }
    )
