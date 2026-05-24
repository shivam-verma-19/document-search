"""
Local reranker using keyword overlap + length bonus.
Replaces amazon.rerank-v1:0 which is not available in ap-south-1.
Same interface: rerank(query, docs) -> list
"""

import os

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


def _score(query: str, doc_text: str) -> float:
    """
    Score a doc against the query using:
    - 80% keyword overlap (how many query words appear in doc)
    - 20% length bonus (longer docs tend to have more context)
    """
    query_words = set(query.lower().split())
    doc_words = doc_text.lower().split()

    overlap = sum(1 for w in doc_words if w in query_words)
    overlap_score = overlap / max(len(query_words), 1)

    length_score = min(len(doc_words) / 200, 1.0)

    return (overlap_score * 0.8) + (length_score * 0.2)


def rerank(query: str, docs: list) -> list:
    """
    Rerank docs by local relevance score.
    Returns docs sorted best-first.
    """
    if not docs:
        return docs

    scored = [(doc, _score(query, doc.page_content)) for doc in docs]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored]
