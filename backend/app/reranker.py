import os

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# Reranking weights — must sum to 1.0.
# Keyword overlap is the primary signal; length provides a small context bonus.
_OVERLAP_WEIGHT = 0.8
_LENGTH_WEIGHT = 0.2

# Documents shorter than this word count receive a proportionally reduced length bonus.
_LENGTH_NORMALISER = 200


def _score(query: str, doc_text: str) -> float:
    """
    Score a doc against the query using:
    - {_OVERLAP_WEIGHT*100:.0f}% keyword overlap (how many query words appear in doc)
    - {_LENGTH_WEIGHT*100:.0f}% length bonus (longer docs tend to have more context)
    """
    query_words = set(query.lower().split())
    doc_words = doc_text.lower().split()

    overlap = sum(1 for w in doc_words if w in query_words)
    overlap_score = overlap / max(len(query_words), 1)

    length_score = min(len(doc_words) / _LENGTH_NORMALISER, 1.0)

    return (overlap_score * _OVERLAP_WEIGHT) + (length_score * _LENGTH_WEIGHT)


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
