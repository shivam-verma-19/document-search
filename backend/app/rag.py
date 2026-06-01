import logging
import re
import time
from typing import Optional

from .cache_service import get_cached_answer, set_cached_answer
from .llm_factory import get_llm_client
from .monitoring import emit_confidence_metric
from .retry import retry_with_backoff
from .s3_vectors_client import get_document, get_documents_by_doc_base_id
from .search_service import hybrid_search, rerank_documents

logger = logging.getLogger(__name__)

# ─── Tunable constants ────────────────────────────────────────────────────────

# Minimum number of retrieved chunks required to use the RAG path.
# Fewer than this falls back to LLM-only to avoid hallucination from a single
# low-confidence chunk.
MIN_DOCS_FOR_RAG: int = 2

# Default number of chunks to retrieve per query.
RAG_TOP_K: int = 5

# Default LLM retry attempts.
LLM_MAX_RETRIES: int = 3

TERSE_SYSTEM = "Reply concise. No filler. No preamble. Facts only."

# ─── Forbidden-query filter ───────────────────────────────────────────────────

# Single-word patterns — matched with \b word boundaries.
_FORBIDDEN_SINGLE = {
    "hack",
    "exploit",
    "malware",
    "ddos",
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
    "vulnerability",
    "cybercrime",
    "cyberattack",
    "password",
    "credentials",
    "secret",
}

# Multi-word patterns — matched with simple substring search after lowercasing.
# \b cannot span spaces, so these are handled separately.
_FORBIDDEN_MULTI = [
    "sql injection",
    "drop table",
    "social engineering",
    "zero-day",
    "api keys",
    "cybersecurity breach",
    "data breach",
    "cyber attack",
]

_forbidden_single_re = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in sorted(_FORBIDDEN_SINGLE, key=len, reverse=True))
    + r")\b",
    flags=re.IGNORECASE,
)


def _is_forbidden(query: str) -> bool:
    """Return True if the query matches any forbidden pattern."""
    lowered = query.lower()
    if _forbidden_single_re.search(query):
        return True
    return any(phrase in lowered for phrase in _FORBIDDEN_MULTI)


# ─── LLM invocation with retry ───────────────────────────────────────────────


def invoke_llm_with_retry(
    prompt: str,
    query: str = "",
    context: str = "",
    max_retries: int = LLM_MAX_RETRIES,
) -> Optional[dict]:
    client = get_llm_client()

    try:
        return retry_with_backoff(
            lambda: client.invoke(prompt=prompt, query=query, context=context),
            max_retries=max_retries,
            base_delay_ms=3000,
        )
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}", exc_info=True)
        return None


# ─── Main pipeline ────────────────────────────────────────────────────────────


def ask_question(query: str) -> str:
    from . import metrics, utils

    start_time = time.time()

    if not query or not query.strip():
        return "Please provide a question."

    query = query.strip()

    if _is_forbidden(query):
        logger.warning(f"Forbidden query pattern: {query[:50]}...")
        emit_confidence_metric(0.0, escalated=True, path="blocked")
        return "This query is not allowed."

    logger.info(f"Processing query: {query[:50]}...")

    cached = get_cached_answer(query)
    if cached:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Cache hit, returned in {elapsed_ms:.0f}ms")
        emit_confidence_metric(1.0, escalated=False, path="cached", source="cached")
        return cached

    docs = hybrid_search(query, k=RAG_TOP_K)
    doc_count = len(docs)
    if docs:
        docs = rerank_documents(query, docs)

    if docs and len(docs) >= MIN_DOCS_FOR_RAG:
        context = "\n".join(d.page_content for d in docs if d.page_content)
        prompt = utils.build_prompt(context, query)
        result = invoke_llm_with_retry(
            prompt=prompt, query=query, context=context, max_retries=LLM_MAX_RETRIES
        )
        if result and result.get("answer"):
            answer = result["answer"]
            set_cached_answer(query, answer)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"RAG response generated in {elapsed_ms:.0f}ms")
            confidence = min(1.0, doc_count / RAG_TOP_K)
            emit_confidence_metric(
                confidence, escalated=False, path="success", source="rag"
            )
            try:
                metrics.log_metrics(query, elapsed_ms, "rag")
            except Exception as e:
                logger.warning(f"Metrics logging failed: {e}")
            return answer

    logger.info(
        f"Falling back to LLM-only (found {len(docs)} docs, need >= {MIN_DOCS_FOR_RAG})"
    )
    prompt = f"{TERSE_SYSTEM}\n\n{query}"
    result = invoke_llm_with_retry(
        prompt=prompt, query=query, context="", max_retries=LLM_MAX_RETRIES
    )
    if result and result.get("answer"):
        answer = f"I couldn't find relevant documents. Based on my knowledge:\n\n{result['answer']}"
        if result and result.get("answer"):
            answer = f"I couldn't find relevant documents. Based on my knowledge:\n\n{result['answer']}"
            if not result["answer"].startswith("Error:"):
                set_cached_answer(query, answer)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Fallback response generated in {elapsed_ms:.0f}ms")
        emit_confidence_metric(0.5, escalated=False, path="fallback", source="llm")
        try:
            metrics.log_metrics(query, elapsed_ms, "llm")
        except Exception as e:
            logger.warning(f"Metrics logging failed: {e}")
        return answer

    logger.error(
        f"All answer generation paths failed after {(time.time()-start_time)*1000:.0f}ms"
    )
    emit_confidence_metric(0.0, escalated=True, path="error")
    return "I'm having trouble generating an answer right now. Please try again in a moment."


# ─── Summarize ────────────────────────────────────────────────────────────────


def summarize_doc(doc_id: str) -> str:
    from . import utils

    if not doc_id or not doc_id.strip():
        return "Please provide a document ID."

    doc_id = doc_id.strip()
    logger.info(f"Summarizing document: {doc_id}")

    docs = []
    if "#" in doc_id:
        doc = get_document(doc_id)
        if doc is not None:
            docs.append(doc)
        else:
            docs = get_documents_by_doc_base_id(doc_id.split("#", 1)[0])
    else:
        docs = get_documents_by_doc_base_id(doc_id)

    if not docs:
        return f"No documents found for ID: {doc_id}"

    context = "\n".join(d.get("_source", {}).get("text", "") for d in docs)
    if not context.strip():
        return "No content available for summarization."

    prompt = f"Summarize the following document concisely. Include key points only:\n\n{context}"
    result = invoke_llm_with_retry(
        prompt=prompt, query="summarize document", context=context, max_retries=2
    )

    if result and result.get("answer"):
        return result["answer"]

    logger.error(f"Failed to summarize document: {doc_id}")
    return "Failed to generate summary. Please try again."
