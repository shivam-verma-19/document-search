import time
import uuid

from backend.app.reranker import rerank
from backend.app.hybrid import BM25Retriever
from backend.app.utils import build_prompt, get_secrets, log_event
from backend.app.cache import get_cache, set_cache
from backend.app.metrics import log_metrics
from backend.app.monitoring import push_metric
from backend.app.evaluation import store_eval
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore


# =========================
# INIT (RUN ONCE)
# =========================
secrets = get_secrets()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

vector_db = PineconeVectorStore(
    index_name="rag-index",
    embedding=embeddings
)

# ⚠️ TEMP: replace later with proper ingestion source
all_docs = vector_db.similarity_search(" ", k=200)
bm25 = BM25Retriever(all_docs)


# =========================
# QUERY REWRITE
# =========================
def rewrite_query(query):
    prompt = f"""
    Rewrite this query to improve retrieval without changing intent:
    {query}
    """
    try:
        return llm.invoke(prompt).content.strip()
    except:
        return query


# =========================
# HYBRID SEARCH
# =========================
def hybrid_search(query, k=5):
    semantic_docs = vector_db.similarity_search(query, k=10)
    keyword_docs = bm25.search(query, k=10)

    seen = set()
    combined = []

    for doc in semantic_docs + keyword_docs:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)

    return combined[:k]


# =========================
# MAIN FUNCTION
# =========================
def ask_question(query):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    print(f"[{request_id}] Query: {query}")

    # =========================
    # CACHE
    # =========================
    cached = get_cache(query)
    if cached:
        latency = int((time.time() - start_time) * 1000)

        push_metric("CacheHit", 1)
        push_metric("Latency", latency)

        log_metrics(query, latency, "cache")
        log_event("query", "cache_hit", latency)

        return cached

    # =========================
    # QUERY REWRITE
    # =========================
    rewritten_query = rewrite_query(query)

    # =========================
    # RETRIEVAL
    # =========================
    docs = hybrid_search(rewritten_query, k=5)

    # =========================
    # RERANK
    # =========================
    docs = rerank(rewritten_query, docs)

    # =========================
    # FALLBACK
    # =========================
    if not docs or len(docs) < 2:
        push_metric("LLMFallback", 1)

        ans = None
        for _ in range(2):
            try:
                ans = llm.invoke(query).content
                break
            except Exception as e:
                print(f"[{request_id}] Retry error: {e}")

        if ans is None:
            ans = "Error generating response."

        latency = int((time.time() - start_time) * 1000)

        final_answer = ("There is no info in the context about this query. Switching to LLM\n", ans)

        set_cache(query, final_answer)

        push_metric("Latency", latency)
        log_metrics(query, latency, "llm")
        log_event("query", "llm_fallback", latency)

        return final_answer

    # =========================
    # CONTEXT
    # =========================
    context = "\n".join([d.page_content for d in docs])
    prompt = build_prompt(context, query)

    # =========================
    # LLM CALL
    # =========================
    ans = None
    for _ in range(2):
        try:
            ans = llm.invoke(prompt).content
            break
        except Exception as e:
            print(f"[{request_id}] Retry error: {e}")

    if ans is None:
        ans = "Error generating response."

    # =========================
    # METRICS
    # =========================
    latency = int((time.time() - start_time) * 1000)

    push_metric("Latency", latency)

    # store generic evaluation (no ground truth)
    store_eval(query, latency, 0)

    log_metrics(query, latency, "rag")
    log_event("query", "success", latency)

    # =========================
    # CACHE STORE
    # =========================
    set_cache(query, ans)

    return ans


# =========================
# SUMMARIZATION
# =========================
def summarize_doc(doc_id):
    docs = vector_db.similarity_search(doc_id, k=10)
    context = "\n".join([d.page_content for d in docs])

    return llm.invoke(f"Summarize:\n{context}").content