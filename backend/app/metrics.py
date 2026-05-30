import os
import time
import uuid

import boto3
from boto3.dynamodb.conditions import Attr

from .monitoring import push_metric

_TABLE_NAME = os.getenv("DYNAMODB_METRICS_TABLE", "rag-metrics")
_METRICS_WINDOW_SECONDS = int(os.getenv("METRICS_WINDOW_SECONDS", str(24 * 3600)))

# FIX: lazy initialisation — do NOT instantiate at module level
_dynamodb = None
_table = None


def _get_table():
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb")
        _table = _dynamodb.Table(_TABLE_NAME)
    return _table


def log_metrics(query, latency, source):
    _get_table().put_item(
        Item={
            "id": str(uuid.uuid4()),
            "query": query,
            "timestamp": int(time.time()),  # FIX: store as int, not str
            "latency": latency,
            "source": source,
        }
    )
    push_metric("QueryLatency", latency, unit="Milliseconds")


def get_metrics(window_seconds: int | None = None):
    if window_seconds is None:
        window_seconds = _METRICS_WINDOW_SECONDS

    since = int(time.time()) - window_seconds  # already int — matches stored type

    items = []

    try:
        response = _get_table().scan(
            ProjectionExpression="id, query, #ts, latency, source",
            FilterExpression=Attr("timestamp").gte(since),
            ExpressionAttributeNames={"#ts": "timestamp"},  # timestamp is reserved
        )

        items.extend(response.get("Items", []))

        last_evaluated_key = response.get("LastEvaluatedKey")

        while last_evaluated_key:
            response = _get_table().scan(
                ProjectionExpression="id, query, #ts, latency, source",
                FilterExpression=Attr("timestamp").gte(since),
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExclusiveStartKey=last_evaluated_key,
            )

            items.extend(response.get("Items", []))
            last_evaluated_key = response.get("LastEvaluatedKey")

    except Exception:
        push_metric("MetricQueryFailure", 1, unit="Count")
        return []

    return items
