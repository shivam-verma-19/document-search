import json

from .ingest import process_s3_upload


def handler(event, context):
    results = []
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        bucket = body["bucket"]
        key = body["key"]
        user = body.get("user")

        results.append(process_s3_upload(bucket, key, user))

    return {"processed": len(results)}
