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


def log_metrics(query, latency, source, user_id: str | None = None):
    item = {
        "id": str(uuid.uuid4()),
        "query": query,
        "timestamp": int(time.time()),
        "latency": latency,
        "source": source,
    }
    if user_id:
        item["user_id"] = user_id
    _get_table().put_item(Item=item)
    push_metric("QueryLatency", latency, unit="Milliseconds")


def get_metrics(window_seconds: int | None = None, user_id: str | None = None):
    if window_seconds is None:
        window_seconds = _METRICS_WINDOW_SECONDS

    since = int(time.time()) - window_seconds  # already int — matches stored type

    items = []

    try:
        filter_expression = Attr("timestamp").gte(since)
        if user_id:
            filter_expression = filter_expression & Attr("user_id").eq(user_id)

        response = _get_table().scan(
            ProjectionExpression="id, query, #ts, latency, source, user_id",
            FilterExpression=filter_expression,
            ExpressionAttributeNames={"#ts": "timestamp"},
        )

        items.extend(response.get("Items", []))

        last_evaluated_key = response.get("LastEvaluatedKey")

        while last_evaluated_key:
            response = _get_table().scan(
                ProjectionExpression="id, query, #ts, latency, source, user_id",
                FilterExpression=filter_expression,
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExclusiveStartKey=last_evaluated_key,
            )

            items.extend(response.get("Items", []))
            last_evaluated_key = response.get("LastEvaluatedKey")

    except Exception:
        push_metric("MetricQueryFailure", 1, unit="Count")
        return []

    return items
