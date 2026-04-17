# backend/app/utils.py

import re

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