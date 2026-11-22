import logging
import os
import time
import uuid

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException

from .monitoring import push_metric

logger = logging.getLogger(__name__)

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
    now = int(time.time())
    item = {
        "id": str(uuid.uuid4()),
        "query": query,
        "timestamp": now,
        "latency": latency,
        "source": source,
        "ttl": now + 90 * 24 * 3600,
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
    last_evaluated_key = None

    try:
        while True:
            if user_id:
                response = _get_table().query(
                    IndexName="user-timestamp-index",
                    KeyConditionExpression=Key("user_id").eq(user_id)
                    & Key("timestamp").gte(since),
                    ProjectionExpression="id, query, #ts, latency, source, user_id",
                    ExpressionAttributeNames={"#ts": "timestamp"},
                    **(
                        {"ExclusiveStartKey": last_evaluated_key}
                        if last_evaluated_key
                        else {}
                    ),
                )
            else:
                response = _get_table().scan(
                    ProjectionExpression="id, query, #ts, latency, source, user_id",
                    FilterExpression=Attr("timestamp").gte(since),
                    ExpressionAttributeNames={"#ts": "timestamp"},
                    **(
                        {"ExclusiveStartKey": last_evaluated_key}
                        if last_evaluated_key
                        else {}
                    ),
                )
            items.extend(response.get("Items", []))
            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

    except (BotoCoreError, ClientError) as exc:
        logger.error("Metrics storage is unavailable", exc_info=exc)
        push_metric("MetricQueryFailure", 1, unit="Count")
        raise HTTPException(
            status_code=503,
            detail="Metrics are temporarily unavailable.",
        ) from exc

    return items
