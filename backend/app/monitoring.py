import logging
import os

import boto3

logger = logging.getLogger(__name__)

_NAMESPACE = os.getenv("METRICS_NAMESPACE", "RAGPlatform")
_cloudwatch_client = None


def get_cloudwatch_client():
    global _cloudwatch_client
    if _cloudwatch_client is None:
        _cloudwatch_client = boto3.client("cloudwatch")  # type: ignore
    return _cloudwatch_client


def push_metric(name: str, value: float, unit: str = "Count", dimensions=None):
    try:
        get_cloudwatch_client().put_metric_data(
            Namespace=_NAMESPACE,
            MetricData=[
                {
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": dimensions or [],
                }
            ],
        )
    except Exception as e:
        logger.debug(f"Metric push failed: {e}")


# ─── CloudWatch confidence metrics ───────────────────────────────────────────


def emit_confidence_metric(
    confidence: float,
    escalated: bool = False,
    path: str = "unknown",
    source: str = "rag",
) -> None:
    push_metric(
        "AnswerConfidence",
        confidence,
        unit="None",
        dimensions=[
            {"Name": "Source", "Value": source},
            {"Name": "Path", "Value": path},
        ],
    )
    if escalated:
        push_metric("EscalationCount", 1)
    push_metric(
        "AnswerGenerated",
        1,
        dimensions=[
            {"Name": "Source", "Value": source},
        ],
    )
