import json
import os

import boto3

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

_bedrock_client = None


def _get_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def get_embedding(text: str) -> list[float]:
    response = _get_client().invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps({"inputText": text}),
    )
    return json.loads(response["body"].read())["embedding"]
