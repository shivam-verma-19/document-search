import json
import os
import re

import boto3

s3 = boto3.client("s3") # type: ignore
BUCKET = os.environ.get("BUCKET_NAME", "rag-pipeline-upload-bucket")


def save_to_s3(file):
    key = file.filename

    s3.upload_fileobj(file.file, BUCKET, key)

    return key


def get_secrets():
    secret_name = os.environ["SECRET_NAME"]

    client = boto3.client("secretsmanager") 

    response = client.get_secret_value(SecretId=secret_name)

    return json.loads(response["SecretString"])


def normalize_text(text: str) -> str:
    """
    Normalize text for better semantic matching
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text: str) -> str:
    """
    Remove unwanted characters
    """
    text = re.sub(r"[^\w\s.,]", "", text)
    return text


def log_event(event, status, latency):
    print(json.dumps({"event": event, "status": status, "latency_ms": latency}))


def build_prompt(context: str, query: str) -> str:
    """
    Standard RAG prompt template
    """
    return f"""
You are an AI assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"There is no information in the context."

Context:
{context}

Question:
{query}

Answer:
"""
