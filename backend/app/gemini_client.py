from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# ─── Complexity classifier ────────────────────────────────────────────────────

_COMPLEX_KEYWORDS = {
    "analyze",
    "analyse",
    "compare",
    "contrast",
    "evaluate",
    "assess",
    "explain why",
    "implications",
    "trade-off",
    "pros and cons",
    "strategy",
    "relationship between",
    "summarize",
    "synthesize",
}

_SIMPLE_KEYWORDS = {
    "who",
    "when",
    "where",
    "what is",
    "define",
    "list",
    "how many",
    "show me",
}

_HEDGING_PHRASES = [
    r"\bnot sure\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bmight\b",
    r"\bcould be\b",
    r"\bverify\b",
]

_HEDGING_RE = [re.compile(p, re.IGNORECASE) for p in _HEDGING_PHRASES]

QueryComplexity = Literal["simple", "complex"]

_client = None


def _get_client():
    """Lazily initialise the Gemini client so module import never triggers a
    network call and key rotation is picked up after a container restart."""
    global _client
    if _client is None:
        from .secrets import get_secret

        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured. Check secrets.")
        _client = genai.Client(api_key=api_key)
    return _client


@dataclass
class ClassifierResult:
    complexity: QueryComplexity
    score: float
    signals: list[str]


@dataclass
class ModelResponse:
    model: str
    text: str
    success: bool
    confidence: float = 0.0
    error: Optional[str] = None


@dataclass
class RouterResult:
    answer: str
    model_used: str
    complexity: str
    confidence: float
    escalated: bool
    attempted: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "answer": self.answer,
            "model_used": self.model_used,
            "complexity": self.complexity,
            "confidence": round(float(self.confidence), 3),
            "escalated": self.escalated,
            "attempted": self.attempted,
            "errors": self.errors,
        }


# ─── Complexity classifier ────────────────────────────────────────────────────


def classify_complexity(query: str, context: str = "") -> ClassifierResult:
    lower = (query or "").lower()
    score = 0.0
    signals = []

    matched_complex = [kw for kw in _COMPLEX_KEYWORDS if kw in lower]
    matched_simple = [kw for kw in _SIMPLE_KEYWORDS if kw in lower]

    if matched_complex:
        score += 0.45
        signals.append("complex keywords")
    if matched_simple and not matched_complex:
        score -= 0.20
        signals.append("simple keywords")
    if len(query.split()) > 15:
        score += 0.20
        signals.append("long query")
    if re.search(r"\b(why|because|although|whereas|since)\b", lower):
        score += 0.15
        signals.append("subordinate clause")
    if len(context) > 1000:
        score += 0.20
        signals.append("long context")

    complexity = "complex" if score >= 0.40 else "simple"
    return ClassifierResult(complexity=complexity, score=score, signals=signals)


# ─── Confidence scorer ────────────────────────────────────────────────────────


def score_confidence(query: str, answer: str) -> float:
    if answer is None:
        return 0.0
    answer = str(answer)
    if not answer.strip():
        return 0.0

    score = 0.75
    answer_words = len(answer.split())
    hedge_hits = sum(1 for p in _HEDGING_RE if p.search(answer))
    score -= hedge_hits * 0.15

    if answer_words < 10:
        score -= 0.40
    elif answer_words < 20:
        score -= 0.10

    if re.search(r"\n\s*[-*•]", answer):
        score += 0.10
    if re.search(r"\b\d{4}\b", answer):
        score += 0.05
    if re.search(r"(no information|cannot answer|not available)", answer, re.I):
        score -= 0.30

    return max(0.0, min(1.0, score))


# ─── Model invoker ────────────────────────────────────────────────────────────


def _invoke_gemini(prompt: str, max_tokens: int = 1024) -> ModelResponse:
    try:
        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        text = (response.text or "").strip()
        return ModelResponse(model=GEMINI_MODEL, text=text, success=True)
    except ValueError as exc:  # missing GEMINI_API_KEY
        logger.error(
            "Gemini config error — check GEMINI_API_KEY in Secrets Manager: %s",
            exc,
            exc_info=True,
        )
        return ModelResponse(model=GEMINI_MODEL, text="", success=False, error=str(exc))
    except Exception as exc:  # API errors, network, etc.
        logger.error("Gemini invocation failed: %s", exc, exc_info=True)
        return ModelResponse(model=GEMINI_MODEL, text="", success=False, error=str(exc))


# ─── Router ───────────────────────────────────────────────────────────────────


def route_and_invoke(prompt: str, query: str = "", context: str = ""):
    """
    One model, two token budgets.
    - Simple query  → 1024 tokens. If low confidence, retry at 2048.
    - Complex query → 2048 tokens. If low confidence, retry at 4096.
    """
    attempted: list[str] = []
    errors: dict[str, str] = {}

    clf = classify_complexity(query, context)
    first_tokens = 1024 if clf.complexity == "simple" else 2048
    retry_tokens = 2048 if clf.complexity == "simple" else 4096

    # ── First attempt ─────────────────────────────────────────────────────────
    resp = _invoke_gemini(prompt, max_tokens=first_tokens)
    attempted.append(resp.model)

    if resp.success and resp.text:
        conf = score_confidence(query, resp.text)
        if conf >= CONFIDENCE_THRESHOLD:
            return RouterResult(
                answer=resp.text,
                model_used=GEMINI_MODEL,
                complexity=clf.complexity,
                confidence=conf,
                escalated=False,
                attempted=attempted,
                errors=errors,
            ).to_dict()
    else:
        errors[resp.model] = resp.error or "unknown"

    # ── Retry with more tokens ─────────────────────────────────────────────────
    retry_label = f"{GEMINI_MODEL}-retry"
    retry = _invoke_gemini(prompt, max_tokens=retry_tokens)
    attempted.append(retry_label)

    if retry.success and retry.text:
        retry_conf = score_confidence(query, retry.text)
        return RouterResult(
            answer=retry.text,
            model_used=retry_label,
            complexity=clf.complexity,
            confidence=retry_conf,
            escalated=True,
            attempted=attempted,
            errors=errors,
        ).to_dict()

    errors[retry_label] = retry.error or "unknown"

    error_summary = f"Errors: {errors} | Attempted: {attempted}"
    logger.error("All model tiers failed. %s", error_summary)

    return RouterResult(
        answer="Error: all model tiers failed to generate a response.",
        model_used="none",
        complexity=clf.complexity,
        confidence=0.0,
        escalated=True,
        attempted=attempted,
        errors=errors,
    ).to_dict()
