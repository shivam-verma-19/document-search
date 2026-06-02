import os
import re

BUCKET = os.environ.get("BUCKET_NAME", "rag-pipeline-upload-bucket")


def normalize_text(text: str) -> str:
    """
    Normalize text for better semantic matching
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip("?!.,;:")
    return text


def clean_text(text: str) -> str:
    """
    Remove unwanted control characters while preserving punctuation.
    """
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def build_prompt(context: str, query: str, sources: list[dict] | None = None) -> str:
    """
    RAG prompt with source-labeled context blocks.

    Each chunk is wrapped with a [Source N | filename | chunk idx] header
    so the LLM can reference where each fact came from.
    """
    if sources:
        # Build labeled context from SearchDocument metadata
        labeled_blocks = []
        for i, src in enumerate(sources, start=1):
            filename = src.get("filename", "unknown")
            chunk_idx = src.get("chunk_index", "?")
            text = src.get("text", "")
            labeled_blocks.append(
                f"[Source {i} | {filename} | chunk {chunk_idx}]\n{text}"
            )
        formatted_context = "\n\n---\n\n".join(labeled_blocks)
    else:
        # Fallback: plain numbered blocks with separator
        blocks = [c.strip() for c in context.split("\n") if c.strip()]
        formatted_context = "\n\n---\n\n".join(
            f"[Source {i}]\n{block}" for i, block in enumerate(blocks, start=1)
        )

    return f"""You are an AI assistant. Answer ONLY using the context below.
If the answer is not in the context, say: "There is no information in the provided documents."

Context:
{formatted_context}

Question:
{query}

Answer:"""
