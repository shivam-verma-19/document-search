"""
Production-grade Cross-Encoder Reranker.

Model:
    Configurable via RERANKER_MODEL env var (default: "BAAI/bge-reranker-base")
    Also supports "cross-encoder/ms-marco-MiniLM-L-6-v2"

Input:
    query: str
    docs: list of SearchDocument or objects with page_content

Output:
    documents sorted by semantic relevance, with automatic lexical fallback
    if deep learning model cannot be loaded or inference fails.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import List

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_model = None
_tokenizer = None
_device = None
_model_failed = False
_model_lock = threading.Lock()


def _lexical_overlap_scores(query: str, texts: List[str]) -> List[float]:
    """Graceful fallback scorer: token overlap with exact phrase matching."""
    q_lower = query.lower().strip()
    query_tokens = set(re.findall(r"\w+", q_lower))
    if not query_tokens:
        return [0.0] * len(texts)

    scores = []
    for text in texts:
        t_lower = text.lower()
        t_tokens = re.findall(r"\w+", t_lower)
        if not t_tokens:
            scores.append(0.0)
            continue
        overlap = sum(1 for tok in t_tokens if tok in query_tokens)
        score = overlap / len(t_tokens)
        # Bonus for exact full phrase match
        if q_lower in t_lower:
            score += 10.0
        scores.append(score)
    return scores


def _load_model():
    global _model, _tokenizer, _device, _model_failed

    if _model is not None or _model_failed:
        return

    with _model_lock:
        if _model is not None or _model_failed:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(
                "Loading Cross-Encoder reranker model: %s on %s", MODEL_NAME, _device
            )

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _model.to(_device)
            _model.eval()

            logger.info("Cross-Encoder reranker loaded successfully")
        except Exception as e:
            logger.warning(
                "Could not load cross-encoder model %s (%s). Falling back to lexical ranking.",
                MODEL_NAME,
                e,
            )
            _model_failed = True


def _predict_scores(
    query: str,
    texts: List[str],
    batch_size: int = 16,
) -> List[float]:
    _load_model()

    if _model_failed or _model is None or _tokenizer is None:
        return _lexical_overlap_scores(query, texts)

    try:
        import torch

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
                if _device is not None:
                    inputs = {k: v.to(_device) for k, v in inputs.items()}

                logits = _model(**inputs).logits
                batch_scores = logits.view(-1).float().cpu().tolist()
                scores.extend(batch_scores)

        return scores
    except Exception as e:
        logger.warning(
            "Cross-encoder inference failed (%s). Using lexical fallback.", e
        )
        return _lexical_overlap_scores(query, texts)


def rerank(query: str, docs: list) -> list:
    """
    Re-rank retrieved documents using cross-encoder model.
    Falls back gracefully to lexical ranking if model fails.
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

    logger.debug("Reranked %s docs", len(docs))
    return [doc for doc, _ in ranked]


def rerank_with_scores(
    query: str,
    docs: list,
):
    """Return ranked tuples of (doc, score) for debugging/eval."""
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
