"""
Retrieval and answer quality evaluation — feedback loop for the RAG pipeline.

Tracks two categories of metrics:

RETRIEVAL QUALITY (per query)
  recall_at_k   — fraction of relevant doc IDs found in top-k results.
                  Requires ground-truth relevant_doc_ids to be supplied.
  mrr           — Mean Reciprocal Rank: 1/rank of first relevant result.
                  Higher = relevant chunk surfaced earlier.

ANSWER QUALITY (per query, LLM-judged)
  faithfulness  — does the answer contain only claims supported by the context?
                  Scored 0.0–1.0 by Gemini.
  relevance     — does the answer actually address the question?
                  Scored 0.0–1.0 by Gemini.

All scores are persisted to DynamoDB (rag-eval table) and pushed to
CloudWatch under the RAGPlatform namespace so they're visible in dashboards
alongside latency.

Usage:
    from .eval import evaluate_retrieval, evaluate_answer, log_eval

    retrieval = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)
    answer    = evaluate_answer(query, context, answer)
    log_eval(query, retrieval, answer)

Env vars:
    DYNAMODB_EVAL_TABLE     DynamoDB table name  (default: "rag-eval")
    EVAL_LLM_ENABLED        "true"/"false" — set False to skip LLM scoring
                            in cost-sensitive environments  (default: "true")
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from backend.app.gemini_client import GEMINI_MODEL, _get_client
from backend.app.monitoring import push_metric

logger = logging.getLogger(__name__)

_TABLE_NAME = os.getenv("DYNAMODB_EVAL_TABLE", "rag-eval")
EVAL_LLM_ENABLED: bool = os.getenv("EVAL_LLM_ENABLED", "true").lower() == "true"

_dynamodb = None
_table = None


def _get_table():
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb")
        _table = _dynamodb.Table(_TABLE_NAME)
    return _table


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0  # fraction of relevant docs retrieved
    mrr: float = 0.0  # mean reciprocal rank
    retrieved_count: int = 0
    relevant_count: int = 0


@dataclass
class AnswerMetrics:
    faithfulness: float = 0.0  # 0–1: claims grounded in context
    relevance: float = 0.0  # 0–1: answer addresses the question
    llm_judged: bool = False  # False when LLM scoring was skipped


@dataclass
class EvalRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time()))
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = field(default_factory=AnswerMetrics)
    ttl: int = field(default_factory=lambda: int(time.time()) + 90 * 24 * 3600)


# ─── Retrieval quality ────────────────────────────────────────────────────────


def evaluate_retrieval(
    query: str,
    retrieved_docs: list,
    relevant_doc_ids: Optional[List[str]] = None,
) -> RetrievalMetrics:
    """
    Compute recall@k and MRR for retrieved documents.

    Args:
        query: User query (for logging).
        retrieved_docs: Ordered list of SearchDocument objects.
        relevant_doc_ids: Ground-truth relevant chunk/doc IDs.
                          If None or empty, recall and MRR are skipped (0.0).

    Returns:
        RetrievalMetrics with recall_at_k and mrr populated.
    """
    retrieved_ids = [
        getattr(doc, "doc_id", "") or getattr(doc, "page_content", "")[:50]
        for doc in retrieved_docs
    ]
    retrieved_count = len(retrieved_ids)

    if not relevant_doc_ids:
        return RetrievalMetrics(retrieved_count=retrieved_count)

    relevant_set = set(relevant_doc_ids)

    # Recall@k — how many relevant docs are in the retrieved set
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
    recall = hits / len(relevant_set) if relevant_set else 0.0

    # MRR — reciprocal rank of the first relevant result
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    logger.debug(
        f"Retrieval eval: recall@{retrieved_count}={recall:.3f}, MRR={mrr:.3f}"
    )
    return RetrievalMetrics(
        recall_at_k=recall,
        mrr=mrr,
        retrieved_count=retrieved_count,
        relevant_count=len(relevant_set),
    )


# ─── Answer quality ───────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are an evaluation assistant.

Rate the FAITHFULNESS of the answer below: does it contain only claims that
are directly supported by the provided context? A score of 1.0 means every
claim is grounded in the context. A score of 0.0 means the answer introduces
information not present in the context (hallucination).

Reply with ONLY a decimal number between 0.0 and 1.0.

Context:
{context}

Answer:
{answer}

Faithfulness score:"""

_RELEVANCE_PROMPT = """You are an evaluation assistant.

Rate the RELEVANCE of the answer to the question: does it directly address
what was asked? A score of 1.0 means the answer fully addresses the question.
A score of 0.0 means the answer is off-topic or does not address the question.

Reply with ONLY a decimal number between 0.0 and 1.0.

Question:
{query}

Answer:
{answer}

Relevance score:"""


def _llm_score(prompt: str) -> float:
    """Call Gemini and parse a float score. Returns 0.5 on any failure."""
    try:
        from google.genai import types

        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=10),
        )
        raw = (response.text or "").strip()
        return max(0.0, min(1.0, float(raw)))
    except Exception as e:
        logger.debug(f"Eval LLM score failed: {e}")
        return 0.5


def evaluate_answer(
    query: str,
    context: str,
    answer: str,
) -> AnswerMetrics:
    """
    Score answer faithfulness and relevance using Gemini as judge.

    If EVAL_LLM_ENABLED is False (cost-saving mode), returns placeholder
    scores of 0.5 with llm_judged=False.

    Args:
        query:   Original user question.
        context: RAG context passed to the LLM (plain text, no prompt wrapper).
        answer:  Generated answer to evaluate.

    Returns:
        AnswerMetrics with faithfulness and relevance scores.
    """
    if not EVAL_LLM_ENABLED:
        return AnswerMetrics(faithfulness=0.5, relevance=0.5, llm_judged=False)

    if not answer or not answer.strip():
        return AnswerMetrics(faithfulness=0.0, relevance=0.0, llm_judged=True)

    faithfulness = _llm_score(
        _FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer[:1000])
    )
    relevance = _llm_score(_RELEVANCE_PROMPT.format(query=query, answer=answer[:1000]))

    logger.debug(
        f"Answer eval: faithfulness={faithfulness:.3f}, relevance={relevance:.3f}"
    )
    return AnswerMetrics(
        faithfulness=faithfulness,
        relevance=relevance,
        llm_judged=True,
    )


# ─── Persistence ──────────────────────────────────────────────────────────────


def log_eval(
    query: str,
    retrieval: RetrievalMetrics,
    answer: AnswerMetrics,
) -> None:
    """
    Persist eval record to DynamoDB and push CloudWatch metrics.

    Failures are swallowed and logged — eval must never break the main pipeline.
    """
    record = EvalRecord(query=query, retrieval=retrieval, answer=answer)

    try:
        _get_table().put_item(
            Item={
                "id": record.id,
                "query": record.query[:500],
                "timestamp": record.timestamp,
                "ttl": record.ttl,
                "recall_at_k": str(round(retrieval.recall_at_k, 4)),
                "mrr": str(round(retrieval.mrr, 4)),
                "retrieved_count": retrieval.retrieved_count,
                "relevant_count": retrieval.relevant_count,
                "faithfulness": str(round(answer.faithfulness, 4)),
                "relevance": str(round(answer.relevance, 4)),
                "llm_judged": answer.llm_judged,
            }
        )
    except ClientError as e:
        logger.warning(f"Eval DynamoDB write failed: {e}")
    except Exception as e:
        logger.warning(f"Eval log failed: {e}")

    # Push to CloudWatch regardless of DynamoDB success
    try:
        push_metric("EvalFaithfulness", answer.faithfulness, unit="None")
        push_metric("EvalRelevance", answer.relevance, unit="None")
        if retrieval.relevant_count > 0:
            push_metric("EvalRecallAtK", retrieval.recall_at_k, unit="None")
            push_metric("EvalMRR", retrieval.mrr, unit="None")
    except Exception as e:
        logger.debug(f"Eval CloudWatch push failed: {e}")


# ─── Convenience: get historical eval records ─────────────────────────────────


def get_eval_records(window_seconds: int = 7 * 24 * 3600) -> list[dict]:
    """
    Return eval records from the last `window_seconds`.
    Used by the /eval endpoint for dashboard visibility.
    """
    from boto3.dynamodb.conditions import Attr

    since = int(time.time()) - window_seconds
    items: list[dict] = []
    try:
        response = _get_table().scan(
            FilterExpression=Attr("timestamp").gte(since),
        )
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        while last_key:
            response = _get_table().scan(
                FilterExpression=Attr("timestamp").gte(since),
                ExclusiveStartKey=last_key,
            )
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
    except Exception as e:
        logger.warning(f"Eval records fetch failed: {e}")
    return items