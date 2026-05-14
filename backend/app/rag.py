import time
import uuid
from dataclasses import dataclass

from . import (
    bedrock_router,
    cache,
    embeddings,
    evaluation,
    metrics,
    monitoring,
    opensearch_client,
    reranker,
    utils,
)
from .hybrid import BM25Retriever


@dataclass
class SearchDocument:
    page_content: str


_forbidden_query_patterns = {
    "hack",
    "exploit",
    "malware",
    "ddos",
    "sql injection",
    "drop table",
}

TERSE_SYSTEM = "Reply concise. No filler. " "No preamble. Facts only."


def _get_bm25(docs):
    return BM25Retriever(docs)


# ─── Query rewrite ────────────────────────────────────────────────────────────


def rewrite_query(query: str):
    if not query:
        return ""

    try:
        result = bedrock_router.route_and_invoke(
            prompt=(f"{TERSE_SYSTEM}\n\n" "Rewrite for retrieval.\n" f"Query: {query}"),
            query=query,
            context="",
        )

        answer = result.get("answer", "").strip()

        if not answer:
            return query

        return answer

    except Exception:
        return query


# ─── Hybrid search ────────────────────────────────────────────────────────────


def hybrid_search(query: str, k: int = 5):
    if not query:
        return []

    try:
        embedding = embeddings.get_embedding(query)

        results = opensearch_client.search_similar(
            embedding,
            k=k,
        )

    except Exception:
        return []

    docs = []

    seen = set()

    for text in results:
        if text and text not in seen:
            seen.add(text)
            docs.append(SearchDocument(page_content=text))

    return docs[:k]


# ─── Main QA ──────────────────────────────────────────────────────────────────


def ask_question(query: str):
    if not query:
        return ""

    lower = query.lower()

    if any(x in lower for x in _forbidden_query_patterns):
        return "Query not allowed"

    cached = cache.get_cache(query)

    if cached:
        return cached

    rewritten = rewrite_query(query)

    docs = hybrid_search(rewritten, k=5)

    try:
        docs = reranker.rerank(rewritten, docs)
    except Exception:
        pass

    # fallback path
    if not docs or len(docs) < 2:
        try:
            routed = bedrock_router.route_and_invoke(
                prompt=f"{TERSE_SYSTEM}\n\n{query}",
                query=query,
                context="",
            )

            answer = (
                "There is no info in the context about this query. "
                "Switching to LLM\n" + routed["answer"]
            )

            cache.set_cache(query, answer)

            return answer

        except Exception:
            return "Error generating response."

    # RAG path
    context = "\n".join(d.page_content for d in docs if d.page_content)

    prompt = utils.build_prompt(
        context,
        query,
    )

    try:
        routed = bedrock_router.route_and_invoke(
            prompt=prompt,
            query=query,
            context=context,
        )

        answer = routed["answer"]

        cache.set_cache(query, answer)

        return answer

    except Exception:
        return "Error generating response."


# ─── Summarization ────────────────────────────────────────────────────────────


def summarize_doc(doc_id: str):
    docs = hybrid_search(doc_id, k=10)

    if not docs:
        return "No content found."

    context = "\n".join(d.page_content for d in docs if d.page_content)

    try:
        routed = bedrock_router.route_and_invoke(
            prompt=("Summarize the following document:\n\n" f"{context}"),
            query="summarize document",
            context=context,
        )

        return routed["answer"]

    except Exception:
        return "Error generating summary."
