import re
import boto3
import json
import os

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
