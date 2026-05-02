import time
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from .cache import get_cache, set_cache
from .evaluation import store_eval
from .hybrid import BM25Retriever
from .metrics import log_metrics
from .monitoring import push_metric
from .reranker import rerank
from .utils import build_prompt, log_event

_forbidden_query_patterns = {
    "hack", "exploit", "malware", "ddos", "sql injection", "drop table"
}


_llm = None
_vector_db = None
_bm25 = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini")
    return _llm


def get_vector_db():
    global _vector_db
    if _vector_db is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vector_db = PineconeVectorStore(index_name="rag-index", embedding=embeddings)
    return _vector_db


def get_bm25():
    global _bm25
    if _bm25 is None:
        try:
            vector_db = get_vector_db()
            docs = vector_db.similarity_search(" ", k=200) or []
        except Exception:
            docs = []

        _bm25 = BM25Retriever(docs)

    return _bm25


def rewrite_query(query: str) -> str:
    if not query:
        return ""

    try:
        response = (
            get_llm()
            .invoke(
                f"Rewrite this query to improve retrieval without changing intent:\n{query}"
            )
            .content
        )
        return str(response).strip()
    except Exception:
        return query


def hybrid_search(query: str, k: int = 5):
    if not query:
        return []

    semantic_docs = []
    keyword_docs = []
    try:
        semantic_docs = get_vector_db().similarity_search(query, k=10) or []
    except Exception:
        pass

    try:
        bm25 = get_bm25()
        keyword_docs = bm25.search(query, k=10) if bm25 else []
    except Exception:
        pass

    seen = set()
    combined = []
    for doc in semantic_docs + keyword_docs:
        content = getattr(doc, "page_content", None)
        if content and content not in seen:
            combined.append(doc)
            seen.add(content)

    return combined[:k]


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
            except Exception as e:
                print(f"[{request_id}] Retry error: {e}")

        if ans is None:
            ans = "Error generating response."

        final_answer = (
            "There is no info in the context about this query. Switching to LLM\n"
            + ans
        )
        latency = int((time.time() - start_time) * 1000)
        set_cache(query, final_answer)
        push_metric("Latency", latency)
        log_metrics(query, latency, "llm")
        log_event("query", "llm_fallback", latency)
        return final_answer

    context = "\n".join(
        [d.page_content for d in docs if getattr(d, "page_content", None)]
    )
    prompt = build_prompt(context, query)

    ans = None
    for _ in range(2):
        try:
            ans = get_llm().invoke(prompt).content
            break
        except Exception as e:
            print(f"[{request_id}] Retry error: {e}")

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
    if not doc_id:
        return ""

    try:
        docs = get_vector_db().similarity_search(doc_id, k=10) or []
    except Exception:
        docs = []

    if not docs:
        return "No content found."

    context = "\n".join(
        [d.page_content for d in docs if getattr(d, "page_content", None)]
    )

    try:
        return get_llm().invoke(f"Summarize:\n{context}").content
    except Exception:
        return "Error generating summary."