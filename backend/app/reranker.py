"""
Production-grade BGE reranker.

Model:
    BAAI/bge-reranker-base

Input:
    query
    retrieved documents

Output:
    documents sorted by semantic relevance
"""

from __future__ import annotations

import logging
import threading
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-reranker-base"

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _load_model():
    global _model
    global _tokenizer

    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return

        logger.info("Loading BGE reranker model: %s", MODEL_NAME)

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        _model.eval()

        logger.info("BGE reranker loaded successfully")


def _predict_scores(
    query: str,
    texts: List[str],
    batch_size: int = 16,
) -> List[float]:

    _load_model()

    scores = []

    with torch.no_grad():

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start : start + batch_size]

            pairs = [[query, text] for text in batch_texts]

            inputs = _tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            logits = _model(**inputs).logits

            batch_scores = logits.view(-1).float().tolist()

            scores.extend(batch_scores)

    return scores


def rerank(query: str, docs: list) -> list:
    """
    Re-rank retrieved documents using BGE cross encoder.
    """

    if not docs:
        return docs

    texts = [getattr(doc, "page_content", "") for doc in docs]

    scores = _predict_scores(query, texts)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.debug(
        "BGE reranked %s docs",
        len(docs),
    )

    return [doc for doc, _ in ranked]


def rerank_with_scores(
    query: str,
    docs: list,
):
    """
    Useful for debugging/evaluation.
    """

    if not docs:
        return []

    texts = [getattr(doc, "page_content", "") for doc in docs]

    scores = _predict_scores(query, texts)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked
