"""
RAGAS-aligned Retrieval and Answer Quality Evaluation.

Tracks the 4 core RAGAS metrics:
1. FAITHFULNESS       — Does the answer contain only claims grounded in the context?
2. ANSWER RELEVANCY   — Does the generated answer directly address the user question?
3. CONTEXT PRECISION  — Are relevant chunks in retrieved context ranked near the top?
4. CONTEXT RECALL     — Does the retrieved context cover all information needed for the answer?

All metrics are persisted to DynamoDB (rag-eval table) and pushed to CloudWatch
under the RAGPlatform namespace for dashboards and alarms.
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

from .gemini_client import GEMINI_MODEL, _get_client
from .monitoring import push_metric

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


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0  # fraction of relevant docs retrieved
    mrr: float = 0.0  # mean reciprocal rank
    context_precision: float = 0.0  # RAGAS Context Precision (0.0–1.0)
    context_recall: float = 0.0  # RAGAS Context Recall (0.0–1.0)
    retrieved_count: int = 0
    relevant_count: int = 0


@dataclass
class AnswerMetrics:
    faithfulness: float = 0.0  # RAGAS Faithfulness (0.0–1.0)
    relevance: float = 0.0  # RAGAS Answer Relevancy (0.0–1.0)
    llm_judged: bool = False


@dataclass
class EvalRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time()))
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = field(default_factory=AnswerMetrics)
    ttl: int = field(default_factory=lambda: int(time.time()) + 90 * 24 * 3600)


# ─── LLM Scoring Helpers ──────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are an evaluation assistant.
Rate the FAITHFULNESS of the answer below: does it contain only claims that
are directly supported by the provided context? A score of 1.0 means every
claim is grounded in the context. A score of 0.0 means the answer introduces
unsupported claims or hallucinated facts.

Reply with ONLY a decimal number between 0.0 and 1.0.

Context:
{context}

Answer:
{answer}

Faithfulness score:"""

_RELEVANCE_PROMPT = """You are an evaluation assistant.
Rate the ANSWER RELEVANCY: does the answer directly, accurately, and
concisely address the user query? A score of 1.0 means fully relevant.
A score of 0.0 means completely irrelevant or divergent.

Reply with ONLY a decimal number between 0.0 and 1.0.

Question:
{query}

Answer:
{answer}

Relevance score:"""

_CONTEXT_PRECISION_PROMPT = """You are an evaluation assistant.
Rate the CONTEXT PRECISION: Do the retrieved context chunks contain useful,
high-signal information with minimal irrelevant noise for answering this question?
A score of 1.0 means highly precise context. A score of 0.0 means purely noise.

Reply with ONLY a decimal number between 0.0 and 1.0.

Question:
{query}

Context:
{context}

Context Precision score:"""

_CONTEXT_RECALL_PROMPT = """You are an evaluation assistant.
Rate the CONTEXT RECALL: Does the retrieved context contain all the essential
facts and information needed to answer the question?
A score of 1.0 means the context covers all needed information.
A score of 0.0 means critical answer information is completely missing.

Reply with ONLY a decimal number between 0.0 and 1.0.

Question:
{query}

Context:
{context}

Context Recall score:"""


def _llm_score(prompt: str) -> float:
    """Call Gemini to obtain a float score from 0.0 to 1.0."""
    try:
        from google.genai import types

        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=10, temperature=0.0),
        )
        raw = (response.text or "").strip()
        return max(0.0, min(1.0, float(raw)))
    except Exception as e:
        logger.debug("Eval LLM score failed: %s", e)
        return 0.5


# ─── Retrieval Evaluation ─────────────────────────────────────────────────────


def evaluate_retrieval(
    query: str,
    retrieved_docs: list,
    relevant_doc_ids: Optional[List[str]] = None,
) -> RetrievalMetrics:
    """
    Compute RAGAS retrieval metrics (Context Precision, Context Recall, Recall@k, MRR).
    """
    retrieved_ids = [
        getattr(doc, "doc_id", "") or getattr(doc, "page_content", "")[:50]
        for doc in retrieved_docs
    ]
    retrieved_count = len(retrieved_ids)

    if not retrieved_docs:
        return RetrievalMetrics(retrieved_count=0)

    # Ground-truth guided evaluation
    if relevant_doc_ids:
        relevant_set = set(relevant_doc_ids)
        hits = 0
        precision_sum = 0.0
        mrr = 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                hits += 1
                precision_sum += hits / rank
                if mrr == 0.0:
                    mrr = 1.0 / rank

        recall = hits / len(relevant_set) if relevant_set else 0.0
        context_precision = (
            precision_sum / len(relevant_set) if relevant_set else 0.0
        )
        context_recall = recall

        logger.debug(
            "Ground-truth retrieval eval: recall=%.3f, precision=%.3f, MRR=%.3f",
            recall,
            context_precision,
            mrr,
        )
        return RetrievalMetrics(
            recall_at_k=recall,
            mrr=mrr,
            context_precision=min(1.0, context_precision),
            context_recall=min(1.0, context_recall),
            retrieved_count=retrieved_count,
            relevant_count=len(relevant_set),
        )

    # Production live query: LLM-judged Context Precision & Context Recall
    if EVAL_LLM_ENABLED and retrieved_docs:
        context_text = "\n".join(
            getattr(d, "page_content", "") for d in retrieved_docs[:5]
        )[:2000]
        c_precision = _llm_score(
            _CONTEXT_PRECISION_PROMPT.format(query=query, context=context_text)
        )
        c_recall = _llm_score(
            _CONTEXT_RECALL_PROMPT.format(query=query, context=context_text)
        )
        return RetrievalMetrics(
            recall_at_k=c_recall,
            mrr=c_precision,
            context_precision=c_precision,
            context_recall=c_recall,
            retrieved_count=retrieved_count,
            relevant_count=0,
        )

    return RetrievalMetrics(
        recall_at_k=1.0,
        mrr=1.0,
        context_precision=1.0,
        context_recall=1.0,
        retrieved_count=retrieved_count,
        relevant_count=0,
    )


# ─── Answer Evaluation ────────────────────────────────────────────────────────


def evaluate_answer(
    query: str,
    context: str,
    answer: str,
) -> AnswerMetrics:
    """
    Score Answer Faithfulness and Relevancy using Gemini judge.
    """
    if not EVAL_LLM_ENABLED:
        return AnswerMetrics(faithfulness=0.5, relevance=0.5, llm_judged=False)

    if not answer or not answer.strip():
        return AnswerMetrics(faithfulness=0.0, relevance=0.0, llm_judged=True)

    faithfulness = _llm_score(
        _FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer[:1000])
    )
    relevance = _llm_score(
        _RELEVANCE_PROMPT.format(query=query, answer=answer[:1000])
    )

    logger.debug(
        "Answer eval: faithfulness=%.3f, relevance=%.3f",
        faithfulness,
        relevance,
    )
    return AnswerMetrics(
        faithfulness=faithfulness,
        relevance=relevance,
        llm_judged=True,
    )


# ─── Persistence & Monitoring ─────────────────────────────────────────────────


def log_eval(
    query: str,
    retrieval: RetrievalMetrics,
    answer: AnswerMetrics,
) -> None:
    """
    Persist RAGAS metrics to DynamoDB and publish to CloudWatch.
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
                "context_precision": str(round(retrieval.context_precision, 4)),
                "context_recall": str(round(retrieval.context_recall, 4)),
                "retrieved_count": retrieval.retrieved_count,
                "relevant_count": retrieval.relevant_count,
                "faithfulness": str(round(answer.faithfulness, 4)),
                "relevance": str(round(answer.relevance, 4)),
                "llm_judged": answer.llm_judged,
            }
        )
    except ClientError as e:
        logger.warning("Eval DynamoDB write failed: %s", e)
    except Exception as e:
        logger.warning("Eval log failed: %s", e)

    # Publish CloudWatch metrics
    try:
        push_metric("EvalFaithfulness", answer.faithfulness, unit="None")
        push_metric("EvalAnswerRelevancy", answer.relevance, unit="None")
        push_metric("EvalContextPrecision", retrieval.context_precision, unit="None")
        push_metric("EvalContextRecall", retrieval.context_recall, unit="None")
        if retrieval.relevant_count > 0:
            push_metric("EvalRecallAtK", retrieval.recall_at_k, unit="None")
            push_metric("EvalMRR", retrieval.mrr, unit="None")
    except Exception as e:
        logger.debug("Eval CloudWatch push failed: %s", e)


def get_eval_records(window_seconds: int = 7 * 24 * 3600) -> list[dict]:
    """Return eval records from the last window_seconds for dashboard reporting."""
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
        logger.warning("Eval records fetch failed: %s", e)
    return items
