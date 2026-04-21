import uuid

import boto3

table = boto3.resource("dynamodb").Table("rag-metrics")


def log_metrics(query, latency, source):
    table.put_item(
        Item={
            "id": str(uuid.uuid4()),
            "query": query,
            "latency": latency,
            "source": source,
        }
    )


def get_metrics():
    res = table.scan()
    return res.get("Items", [])
