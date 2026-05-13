import os

USE_BEDROCK = os.getenv("USE_BEDROCK", "false") == "true"


def openai_embedding(text: str):
    from openai import OpenAI

    client = OpenAI()

    return (
        client.embeddings.create(model="text-embedding-3-small", input=text)
        .data[0]
        .embedding
    )


def bedrock_embedding(text: str):
    import json

    import boto3

    client = boto3.client("bedrock-runtime")  # type: ignore

    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v1", body=json.dumps({"inputText": text})
    )

    return json.loads(response["body"].read())["embedding"]


def get_embedding(text: str):
    return bedrock_embedding(text) if USE_BEDROCK else openai_embedding(text)
