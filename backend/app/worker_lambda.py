import json
import logging

from .processor import process_file_from_s3

logger = logging.getLogger(__name__)


def handler(event, context):
    batch_item_failures = []

    for record in event.get("Records", []):
        message_id = record["messageId"]
        try:
            body = json.loads(record["body"])
            result = process_file_from_s3(body["bucket"], body["key"])
            if result.get("status") == "error":
                batch_item_failures.append({"itemIdentifier": message_id})
        except Exception:
            logger.exception("Failed to process SQS message %s", message_id)
            batch_item_failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": batch_item_failures}
