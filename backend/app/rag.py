import time
import uuid
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

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

_llm = None
_bm25 = None
_vector_db = None


def get_llm():
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini")  # type: ignore

    return _llm


def get_bm25():
    global _bm25

    if _bm25 is None:
        _bm25 = BM25Retriever([])

    return _bm25


def rewrite_query(query: str) -> str:
    if not query:
        return ""

    try:
        response = (
            get_llm()
            .invoke(
                f"Rewrite this query to improve retrieval "
                f"without changing intent:\n{query}"
            )
            .content
        )

        return str(response).strip()

    except Exception:
        return query


def hybrid_search(query: str, k: int = 5):
    if not query:
        return []

    try:
        embedding = get_embedding(query)

        texts = search_similar(embedding, k=k)

    except Exception:
        texts = []

    docs = [SearchDocument(page_content=t) for t in texts]

    return docs


def ask_question(query: str):
    lower = query.lower()

    if any(pattern in lower for pattern in _forbidden_query_patterns):
        return "Query not allowed"

    if not query:
        return ""

    request_id = str(uuid.uuid4())
    start_time = time.time()

    print(f"[{request_id}] Query: {query}")

    cached = get_cache(query)

    if cached:
        latency = int((time.time() - start_time) * 1000)

        push_metric("CacheHit", 1)
        push_metric("Latency", latency)

        log_metrics(query, latency, "cache")
        log_event("query", "cache_hit", latency)

        return cached

    rewritten_query = rewrite_query(query)

    docs = hybrid_search(rewritten_query, k=5)

    try:
        docs = rerank(rewritten_query, docs)

    except Exception:
        pass

    if not docs or len(docs) < 2:
        push_metric("LLMFallback", 1)

        ans = None

        for _ in range(2):
            try:
                response = get_llm().invoke(query).content
                ans = str(response)
                break

            except Exception as exc:
                print(f"[{request_id}] Retry error: {exc}")

        if ans is None:
            ans = "Error generating response."

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

    context = "\n".join([d.page_content for d in docs if d.page_content])

    prompt = build_prompt(context, query)

    ans = None

    for _ in range(2):
        try:
            ans = get_llm().invoke(prompt).content
            break

        except Exception as exc:
            print(f"[{request_id}] Retry error: {exc}")

    if ans is None:
        ans = "Error generating response."

    latency = int((time.time() - start_time) * 1000)

    push_metric("Latency", latency)

    store_eval(query, latency, 0)

    log_metrics(query, latency, "rag")
    log_event("query", "success", latency)

    set_cache(query, ans)

    return ans


def summarize_doc(doc_id: str):
    docs = hybrid_search(doc_id, k=10)

    if not docs:
        return "No content found."

    context = "\n".join([d.page_content for d in docs if d.page_content])

    try:
        return get_llm().invoke(f"Summarize:\n{context}").content

    except Exception:
        return "Error generating summary."
