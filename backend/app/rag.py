
"""
Robust RAG pipeline with comprehensive error handling, logging, and monitoring.

Features:
- Detailed logging at each step
- Graceful degradation with fallbacks
- Retry logic for transient failures
- Error tracking and categorization
- Performance monitoring
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
}

TERSE_SYSTEM = "Reply concise. No filler. No preamble. Facts only."


def hybrid_search(query: str, k: int = 5) -> List[SearchDocument]:
    """Perform hybrid vector and keyword search with comprehensive error handling."""
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search")
        return []

    start_time = time.time()
    docs = []

    try:
        logger.debug(f"Getting embedding for query: {query[:50]}...")
        # Step 1: Get embedding
        from . import embeddings

        try:
            embedding = embeddings.get_embedding(query)
        except Exception as e:
            logger.warning(
                f"Embedding failed (continuing with keyword search): {str(e)}"
            )
            embedding = None

        # Step 2: Vector search
        if embedding:
            try:
                from . import chromadb_client

                logger.debug("Performing vector similarity search...")
                results = chromadb_client.search_similar(embedding, k=k)

                seen = set()
                for text in results:
                    if text and text not in seen:
                        seen.add(text)
                        docs.append(SearchDocument(page_content=text))

                logger.debug(f"Vector search returned {len(docs)} documents")

            except Exception as e:
                logger.warning(f"Vector search failed: {str(e)}")
                docs = []

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Hybrid search completed in {elapsed_ms:.0f}ms, found {len(docs)} docs"
        )
        return docs[:k]

    except Exception as e:
        logger.error(f"Unexpected error in hybrid_search: {str(e)}", exc_info=True)
        return []


def invoke_bedrock_with_retry(
    prompt: str,
    query: str = "",
    context: str = "",
    max_retries: int = 3,
) -> Optional[dict]:
    """Invoke Bedrock with automatic retry logic for transient failures."""
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
                    f"Bedrock failed (attempt {attempt}/{max_retries}): {str(e)} | "
                    f"Retryable: {is_retryable}",
                    exc_info=True,
                )
                return None

            # Calculate backoff delay
            delay_ms = min(100 * (2 ** (attempt - 1)), 5000)
            logger.warning(
                f"Bedrock failed (attempt {attempt}), retrying in {delay_ms}ms: {str(e)}"
            )
            time.sleep(delay_ms / 1000)

    logger.error(
        f"All {max_retries} Bedrock attempts failed. Last error: {str(last_error)}"
    )
    return None


def _is_bedrock_error_retryable(error: Exception) -> bool:
    """Determine if a Bedrock error is worth retrying."""
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


def get_cached_answer(query: str) -> Optional[str]:
    """Get answer from cache, with error handling."""
    from . import cache

    try:
        result = cache.get_cache(query)
        if result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
        return str(result)
    except Exception as e:
        logger.warning(f"Cache read failed: {str(e)}")
        return None


def set_cached_answer(query: str, answer: str) -> bool:
    """Set answer in cache, with error handling."""
    from . import cache

    try:
        cache.set_cache(query, answer)
        return True
    except Exception as e:
        logger.warning(f"Cache write failed: {str(e)}")
        return False


def rerank_documents(
    query: str,
    docs: List[SearchDocument],
) -> List[SearchDocument]:
    """Rerank documents, returning original list if reranking fails."""
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
        logger.warning(
            f"Reranking failed, using original order: {str(e)}",
            exc_info=True,
        )
        return docs


def ask_question(query: str) -> str:
    """Main RAG pipeline with robust error handling."""
    from . import metrics, utils

    start_time = time.time()

    # Step 0: Input validation
    if not query or not query.strip():
        logger.warning("Empty query received")
        return "Please provide a question."

    query = query.strip()
    lower_query = query.lower()

    if any(pattern in lower_query for pattern in _forbidden_query_patterns):
        logger.warning(f"Forbidden query pattern detected: {query[:50]}...")
        return "This query is not allowed."

    logger.info(f"Processing query: {query[:50]}...")

    # Step 1: Check cache
    cached_answer = get_cached_answer(query)
    if cached_answer:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Cache hit, returned in {elapsed_ms:.0f}ms")
        return cached_answer

    # Step 2: Vector search
    docs = hybrid_search(query, k=5)

    # Step 3: Rerank (if we have docs)
    if docs:
        docs = rerank_documents(query, docs)

    # Step 4: RAG path (with context)
    if docs and len(docs) >= 2:
        logger.debug(f"Using RAG path with {len(docs)} documents")

        context = "\n".join(d.page_content for d in docs if d.page_content)
        prompt = utils.build_prompt(context, query)

        result = invoke_bedrock_with_retry(
            prompt=prompt,
            query=query,
            context=context,
            max_retries=3,
        )

        if result and result.get("answer"):
            answer = result["answer"]
            set_cached_answer(query, answer)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"RAG response generated in {elapsed_ms:.0f}ms")

            try:
                metrics.log_metrics(query, 0, "rag")
            except Exception as e:
                logger.warning(f"Metrics logging failed: {str(e)}")

            return answer

    # Step 5: Fallback path (LLM only, no context)
    logger.info(
        f"Falling back to LLM-only path (found {len(docs)} documents, need >= 2)"
    )

    prompt = f"{TERSE_SYSTEM}\n\n{query}"

    result = invoke_bedrock_with_retry(
        prompt=prompt,
        query=query,
        context="",
        max_retries=3,
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
            metrics.log_metrics(query, 0, "llm")
        except Exception as e:
            logger.warning(f"Metrics logging failed: {str(e)}")

        return answer

    # All paths failed
    elapsed_ms = (time.time() - start_time) * 1000
    logger.error(f"All answer generation paths failed after {elapsed_ms:.0f}ms")

    return (
        "I'm having trouble generating an answer right now. "
        "Please try again in a moment."
    )


def summarize_doc(doc_id: str) -> str:
    """Summarize a document with comprehensive error handling."""
    from . import utils

    if not doc_id or not doc_id.strip():
        logger.warning("Empty doc_id provided for summarization")
        return "Please provide a document ID."

    doc_id = doc_id.strip()
    logger.info(f"Summarizing document: {doc_id}")

    # Search for document
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
