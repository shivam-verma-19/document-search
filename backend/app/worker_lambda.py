import json

from .ingest import upload_file_to_s3


def handler(event, context):
    results = []
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        bucket = body["bucket"]
        key = body["key"]
        user = body.get("user")

        results.append(upload_file_to_s3(key, user))

    return {"processed": len(results)}
