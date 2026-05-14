import time
import uuid
from dataclasses import dataclass

from .bedrock_router import route_and_invoke
from .cache import get_cache, set_cache
from .embeddings import get_embedding
from .evaluation import store_eval
from .hybrid import BM25Retriever
from .metrics import log_metrics
from .monitoring import push_metric
from .opensearch_client import search_similar
from .reranker import rerank
from .utils import build_prompt, log_event


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

# ─── Terse system prompt ──────────────────────────────────────────────────────
# Prepended to every LLM call — caveman-style token reduction.
# Cuts output tokens ~30-50% with no accuracy loss.
TERSE_SYSTEM = (
    "Reply concise. No filler. No preamble. " "Facts only. Use fragments where clear."
)

_bm25 = None


def _get_bm25(docs: list) -> BM25Retriever:
    return BM25Retriever(docs)


# ─── Query rewrite ────────────────────────────────────────────────────────────


def rewrite_query(query: str) -> str:
    if not query:
        return ""
    try:
        result = route_and_invoke(
            prompt=(
                f"{TERSE_SYSTEM}\n\n"
                "Rewrite for document retrieval. Same intent. "
                f"Return rewritten query only.\n\nQuery: {query}"
            ),
            query=query,
            context="",  # no doc context for rewriting — keeps it simple
        )
        rewritten = result["answer"].strip()
        return rewritten if rewritten else query
    except Exception:
        return query


# ─── Hybrid retrieval ─────────────────────────────────────────────────────────


def hybrid_search(query: str, k: int = 5):
    if not query:
        return []

    # Vector search (OpenSearch)
    vector_docs: list[SearchDocument] = []
    try:
        embedding = get_embedding(query)
        texts = search_similar(embedding, k=k)
        vector_docs = [SearchDocument(page_content=t) for t in texts]
    except Exception:
        pass

    # BM25 re-rank/supplement over the same corpus
    bm25_docs: list[SearchDocument] = []
    if vector_docs:
        bm25 = _get_bm25(vector_docs)
        bm25_results = bm25.search(query, k=k)
        bm25_docs = [
            SearchDocument(page_content=getattr(d, "page_content", ""))
            for d in bm25_results
        ]

    # Merge & deduplicate, preserving vector order
    seen: set[str] = set()
    merged: list[SearchDocument] = []
    for doc in vector_docs + bm25_docs:
        key = doc.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(doc)
        if len(merged) >= k:
            break

    return merged


# ─── Main Q&A entry point ─────────────────────────────────────────────────────


def ask_question(query: str):
    lower = query.lower()

    if any(pattern in lower for pattern in _forbidden_query_patterns):
        return "Query not allowed"

    if not query:
        return ""

    request_id = str(uuid.uuid4())
    start_time = time.time()
    print(f"[{request_id}] Query: {query}")

    # ── Cache hit ──────────────────────────────────────────────────────────────
    cached = get_cache(query)
    if cached:
        latency = int((time.time() - start_time) * 1000)
        push_metric("CacheHit", 1)
        push_metric("Latency", latency)
        log_metrics(query, latency, "cache")
        log_event("query", "cache_hit", latency)
        return cached

    # ── Query rewrite ──────────────────────────────────────────────────────────
    rewritten_query = rewrite_query(query)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    docs = hybrid_search(rewritten_query, k=5)
    try:
        docs = rerank(rewritten_query, docs)
    except Exception:
        pass

    # ── LLM fallback (no context found) ───────────────────────────────────────
    if not docs or len(docs) < 2:
        push_metric("LLMFallback", 1)

        routed = None
        routed = None
        try:
            routed = route_and_invoke(
                prompt=f"{TERSE_SYSTEM}\n\n{query}",
                query=query,
                context="",
            )
            print(
                f"[{request_id}] Fallback — model={routed['model_used']} "
                f"complexity={routed['complexity']} "
                f"confidence={routed['confidence']:.2f} "
                f"escalated={routed['escalated']}"
            )
        except Exception as exc:
            print(f"[{request_id}] Fallback error: {exc}")

        ans = routed["answer"] if routed else "Error generating response."
        final_answer = (
            "There is no info in the context about this query. "
            "Switching to LLM\n" + ans
        )

        latency = int((time.time() - start_time) * 1000)
        set_cache(query, final_answer)
        push_metric("Latency", latency)
        log_metrics(query, latency, "llm")
        log_event("query", "llm_fallback", latency)
        return final_answer

    # ── RAG answer via intelligent router ─────────────────────────────────────
    context = "\n".join([d.page_content for d in docs if d.page_content])
    base_prompt = build_prompt(context, query)
    prompt = f"{TERSE_SYSTEM}\n\n{base_prompt}"

    routed = None
    try:
        routed = route_and_invoke(
            prompt=prompt,
            query=query,
            context=context,
        )
        print(
            f"[{request_id}] RAG — model={routed['model_used']} "
            f"complexity={routed['complexity']} "
            f"confidence={routed['confidence']:.2f} "
            f"escalated={routed['escalated']} "
            f"attempted={routed['attempted']}"
        )
    except Exception as exc:
        print(f"[{request_id}] RAG error: {exc}")

    ans = routed["answer"] if routed else "Error generating response."

    latency = int((time.time() - start_time) * 1000)
    push_metric("Latency", latency)
    store_eval(query, latency, 0)
    log_metrics(query, latency, "rag")
    log_event("query", "success", latency)
    set_cache(query, ans)

    return ans


# ─── Document summarization ───────────────────────────────────────────────────


def summarize_doc(doc_id: str):
    docs = hybrid_search(doc_id, k=10)
    if not docs:
        return "No content found."

    context = "\n".join([d.page_content for d in docs if d.page_content])
    # Summarization is always complex — classifier will pick Claude Sonnet
    try:
        routed = route_and_invoke(
            prompt=f"Summarize the following document:\n\n{context}",
            query="summarize synthesize the document",
            context=context,
        )
        return routed["answer"]
    except Exception:
        return "Error generating summary."
