import json

from .processor import process_file_from_s3


def handler(event, context):
    results = []

    for record in event.get("Records", []):
        body = json.loads(record["body"])

        bucket = body["bucket"]
        key = body["key"]

        result = process_file_from_s3(bucket, key)
        results.append(result)

    return {"processed": len(results)}
