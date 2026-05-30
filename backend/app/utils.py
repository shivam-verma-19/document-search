import json
import os
import re

BUCKET = os.environ.get("BUCKET_NAME", "rag-pipeline-upload-bucket")


def normalize_text(text: str) -> str:
    """
    Normalize text for better semantic matching
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text: str) -> str:
    """
    Remove unwanted control characters while preserving punctuation.
    """
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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
