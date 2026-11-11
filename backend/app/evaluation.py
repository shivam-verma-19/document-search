"""
Legacy evaluation module maintaining backward compatibility for store_eval
while integrating with the RAGAS evaluation harness in eval.py.
"""

from __future__ import annotations

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
    _get_table().put_item(
        Item={"query": query, "latency": latency, "precision": precision}
    )
