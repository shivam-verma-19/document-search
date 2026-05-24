"""
RAG pipeline with:
- Fixed cache bug (str(None) -> "None" false hit)
- Real hybrid search: FAISS vector + BM25 keyword with RRF fusion
- Local reranker (no Bedrock rerank API needed)
- ap-south-1 compatible throughout
- Router error logging: model failures now surface in CloudWatch
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


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
    "api keys",
    "secret",
    "password",
    "credentials",
    "vulnerability",
    "attack",
    "phishing",
    "ransomware",
    "trojan",
    "worm",
    "spyware",
    "keylogger",
    "backdoor",
    "rootkit",
    "botnet",
    "cryptojacking",
    "social engineering",
    "zero-day",
    "cybercrime",
    "cyberattack",
    "cybersecurity breach",
    "data breach",
}

TERSE_SYSTEM = "Reply concise. No filler. No preamble. Facts only."


# ─── RRF fusion ───────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    list1: List[SearchDocument],
    list2: List[SearchDocument],
    k: int = 5,
    rrf_k: int = 60,
) -> List[SearchDocument]:
    scores: dict[str, float] = {}
    doc_map: dict[str, SearchDocument] = {}

    for rank, doc in enumerate(list1, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(list2, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:k]]


# ─── Hybrid search ────────────────────────────────────────────────────────────


def hybrid_search(query: str, k: int = 5) -> List[SearchDocument]:
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search")
        return []

    start_time = time.time()
    vector_docs: List[SearchDocument] = []
    bm25_docs: List[SearchDocument] = []

    try:
        from . import embeddings, faiss_client

        embedding = embeddings.get_embedding(query)
        results = faiss_client.search_similar(embedding, k=k)
        seen: set[str] = set()
        for text in results:
            if text and text not in seen:
                seen.add(text)
                vector_docs.append(SearchDocument(page_content=text))
        logger.debug(f"Vector search returned {len(vector_docs)} docs")
    except Exception as e:
        logger.warning(f"Vector search failed (continuing with keyword only): {e}")

    try:
        from . import faiss_client
        from .hybrid import BM25Retriever

        all_texts = faiss_client.get_all_documents()
        all_docs = [SearchDocument(page_content=t) for t in all_texts]
        bm25_docs = BM25Retriever(all_docs).search(query, k=k)
        logger.debug(f"BM25 search returned {len(bm25_docs)} docs")
    except Exception as e:
        logger.warning(f"BM25 search failed (continuing with vector only): {e}")

    fused = _reciprocal_rank_fusion(vector_docs, bm25_docs, k=k)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Hybrid search (RRF) completed in {elapsed_ms:.0f}ms, {len(fused)} docs"
    )
    return fused


# ─── Bedrock invocation ───────────────────────────────────────────────────────


def invoke_bedrock_with_retry(
    prompt: str,
    query: str = "",
    context: str = "",
    max_retries: int = 3,
) -> Optional[dict]:
    from . import bedrock_router

    attempt = 0
    last_error = None

    while attempt < max_retries:
        attempt += 1
        try:
            logger.debug(f"Bedrock invocation (attempt {attempt}/{max_retries})...")
            start_time = time.time()

            result = bedrock_router.route_and_invoke(
                prompt=prompt,
                query=query,
                context=context,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # ── FIX: log when router returns an error result (no exception raised) ──
            if result.get("model_used") == "none":
                logger.error(
                    f"Bedrock router: all model tiers failed | "
                    f"Attempted: {result.get('attempted')} | "
                    f"Errors: {result.get('errors')} | "
                    f"elapsed={elapsed_ms:.0f}ms"
                )
                return None  # treat as failure so caller falls through correctly

            logger.info(
                f"Bedrock succeeded in {elapsed_ms:.0f}ms | "
                f"Model: {result.get('model_used')} | "
                f"Confidence: {result.get('confidence', 0):.2f}"
            )
            return result

        except Exception as e:
            last_error = e
            is_retryable = _is_bedrock_error_retryable(e)

            if not is_retryable or attempt >= max_retries:
                logger.error(
                    f"Bedrock failed (attempt {attempt}/{max_retries}): {e} | "
                    f"Retryable: {is_retryable}",
                    exc_info=True,
                )
                return None

            delay_ms = min(100 * (2 ** (attempt - 1)), 5000)
            logger.warning(
                f"Bedrock failed (attempt {attempt}), retrying in {delay_ms}ms: {e}"
            )
            time.sleep(delay_ms / 1000)

    logger.error(f"All {max_retries} Bedrock attempts failed. Last: {last_error}")
    return None


def _is_bedrock_error_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    retryable_keywords = [
        "timeout",
        "throttling",
        "rate limit",
        "503",
        "502",
        "500",
        "unavailable",
    ]
    return any(keyword in error_str for keyword in retryable_keywords)


# ─── Cache ────────────────────────────────────────────────────────────────────


def get_cached_answer(query: str) -> Optional[str]:
    from . import cache

    try:
        result = cache.get_cache(query)
        if result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return str(result)
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None


def set_cached_answer(query: str, answer: str) -> bool:
    from . import cache

    try:
        cache.set_cache(query, answer)
        return True
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
        return False


# ─── Reranker ────────────────────────────────────────────────────────────────


def rerank_documents(query: str, docs: List[SearchDocument]) -> List[SearchDocument]:
    from . import reranker

    if not docs:
        return docs
    try:
        logger.debug(f"Reranking {len(docs)} documents...")
        start_time = time.time()
        reranked = reranker.rerank(query, docs)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Reranking completed in {elapsed_ms:.0f}ms")
        return reranked
    except Exception as e:
        logger.warning(f"Reranking failed, using original order: {e}", exc_info=True)
        return docs


# ─── Main pipeline ────────────────────────────────────────────────────────────


def ask_question(query: str) -> str:
    from . import metrics, utils

    start_time = time.time()

    if not query or not query.strip():
        logger.warning("Empty query received")
        return "Please provide a question."

    query = query.strip()
    lower_query = query.lower()

    if any(pattern in lower_query for pattern in _forbidden_query_patterns):
        logger.warning(f"Forbidden query pattern detected: {query[:50]}...")
        return "This query is not allowed."

    logger.info(f"Processing query: {query[:50]}...")

    cached_answer = get_cached_answer(query)
    if cached_answer:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Cache hit, returned in {elapsed_ms:.0f}ms")
        return cached_answer

    docs = hybrid_search(query, k=5)

    if docs:
        docs = rerank_documents(query, docs)

    if docs and len(docs) >= 2:
        logger.debug(f"Using RAG path with {len(docs)} documents")
        context = "\n".join(d.page_content for d in docs if d.page_content)
        prompt = utils.build_prompt(context, query)

        result = invoke_bedrock_with_retry(
            prompt=prompt, query=query, context=context, max_retries=3
        )

        if result and result.get("answer"):
            answer = result["answer"]
            set_cached_answer(query, answer)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"RAG response generated in {elapsed_ms:.0f}ms")
            try:
                metrics.log_metrics(query, elapsed_ms, "rag")
            except Exception as e:
                logger.warning(f"Metrics logging failed: {e}")
            return answer

    logger.info(f"Falling back to LLM-only path (found {len(docs)} docs, need >= 2)")
    prompt = f"{TERSE_SYSTEM}\n\n{query}"

    result = invoke_bedrock_with_retry(
        prompt=prompt, query=query, context="", max_retries=3
    )

    if result and result.get("answer"):
        answer = (
            f"I couldn't find relevant documents. Based on my knowledge:\n\n"
            f"{result['answer']}"
        )
        set_cached_answer(query, answer)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Fallback response generated in {elapsed_ms:.0f}ms")
        try:
            metrics.log_metrics(query, elapsed_ms, "llm")
        except Exception as e:
            logger.warning(f"Metrics logging failed: {e}")
        return answer

    elapsed_ms = (time.time() - start_time) * 1000
    logger.error(f"All answer generation paths failed after {elapsed_ms:.0f}ms")
    return "I'm having trouble generating an answer right now. Please try again in a moment."


# ─── Summarize ────────────────────────────────────────────────────────────────


def summarize_doc(doc_id: str) -> str:
    from . import utils

    if not doc_id or not doc_id.strip():
        logger.warning("Empty doc_id provided for summarization")
        return "Please provide a document ID."

    doc_id = doc_id.strip()
    logger.info(f"Summarizing document: {doc_id}")

    docs = hybrid_search(doc_id, k=10)

    if not docs:
        logger.warning(f"No documents found for doc_id: {doc_id}")
        return f"No documents found for ID: {doc_id}"

    context = "\n".join(d.page_content for d in docs if d.page_content)

    if not context.strip():
        logger.warning(f"No extractable content found for doc_id: {doc_id}")
        return "No content available for summarization."

    prompt = (
        "Summarize the following document concisely. Include key points only:\n\n"
        f"{context}"
    )

    result = invoke_bedrock_with_retry(
        prompt=prompt,
        query="summarize document",
        context=context,
        max_retries=2,
    )

    if result and result.get("answer"):
        logger.info("Document summarization successful")
        return result["answer"]

    logger.error(f"Failed to summarize document: {doc_id}")
    return "Failed to generate summary. Please try again."
