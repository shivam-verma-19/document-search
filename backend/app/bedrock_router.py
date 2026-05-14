from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import boto3
import requests

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

CLAUDE_MODEL_ID = os.getenv(
    "BEDROCK_CLAUDE_MODEL_ID",
    "anthropic.claude-sonnet-4-5",
)

LLAMA_MODEL_ID = os.getenv(
    "BEDROCK_LLAMA_MODEL_ID",
    "meta.llama3-8b-instruct-v1:0",
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

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

# ─── Data classes ─────────────────────────────────────────────────────────────

QueryComplexity = Literal["simple", "complex"]


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


# ─── Lazy Bedrock client ──────────────────────────────────────────────────────

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client

    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
        )

    return _bedrock_client


# ─── Complexity classifier ────────────────────────────────────────────────────


def classify_complexity(
    query: str,
    context: str = "",
) -> ClassifierResult:
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

    token_count = len(query.split())

    if token_count > 15:
        score += 0.20
        signals.append("long query")

    if re.search(
        r"\b(why|because|although|whereas|since)\b",
        lower,
    ):
        score += 0.15
        signals.append("subordinate clause")

    if len(context) > 1000:
        score += 0.20
        signals.append("long context")

    complexity = "complex" if score >= 0.40 else "simple"

    return ClassifierResult(
        complexity=complexity,
        score=score,
        signals=signals,
    )


# ─── Confidence scorer ────────────────────────────────────────────────────────


def score_confidence(query: str, answer: str) -> float:
    if not answer or not answer.strip():
        return 0.0

    score = 0.75

    answer_words = len(answer.split())

    hedge_hits = sum(1 for p in _HEDGING_RE if p.search(answer))

    score -= hedge_hits * 0.15

    if answer_words < 10:
        score -= 0.40
    elif answer_words < 20:
        score -= 0.15

    if re.search(r"\n[-*•]", answer):
        score += 0.10

    if re.search(r"\b\d{4}\b", answer):
        score += 0.05

    if re.search(
        r"(no information|cannot answer|not available)",
        answer,
        re.I,
    ):
        score -= 0.30

    return max(0.0, min(1.0, score))


# ─── Model invokers ───────────────────────────────────────────────────────────


def _invoke_claude(
    prompt: str,
    max_tokens: int = 1024,
) -> ModelResponse:
    try:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }
        )

        resp = _get_bedrock_client().invoke_model(
            modelId=CLAUDE_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        text = json.loads(resp["body"].read())["content"][0]["text"].strip()

        return ModelResponse(
            model="claude-sonnet",
            text=text,
            success=True,
        )

    except Exception as exc:
        return ModelResponse(
            model="claude-sonnet",
            text="",
            success=False,
            error=str(exc),
        )


def _invoke_llama(prompt: str) -> ModelResponse:
    try:
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": 1024,
                "temperature": 0.3,
            }
        )

        resp = _get_bedrock_client().invoke_model(
            modelId=LLAMA_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        text = json.loads(resp["body"].read()).get("generation", "").strip()

        return ModelResponse(
            model="llama3-bedrock",
            text=text,
            success=True,
        )

    except Exception as exc:
        return ModelResponse(
            model="llama3-bedrock",
            text="",
            success=False,
            error=str(exc),
        )


def _invoke_ollama(prompt: str) -> ModelResponse:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )

        resp.raise_for_status()

        text = resp.json().get("response", "").strip()

        if not text:
            raise ValueError("Empty response")

        return ModelResponse(
            model=f"ollama-{OLLAMA_MODEL}",
            text=text,
            success=True,
        )

    except Exception as exc:
        return ModelResponse(
            model=f"ollama-{OLLAMA_MODEL}",
            text="",
            success=False,
            error=str(exc),
        )


# ─── Router ───────────────────────────────────────────────────────────────────


def route_and_invoke(
    prompt: str,
    query: str = "",
    context: str = "",
):
    attempted = []
    errors = {}

    clf = classify_complexity(query, context)

    # ─── SIMPLE ───────────────────────────────────────────────────────────────

    # ─── SIMPLE ───────────────────────────────────────────────────────────────

    if clf.complexity == "simple":
        llama = _invoke_llama(prompt)

        attempted.append(llama.model)

        if llama.success and llama.text:
            conf = score_confidence(query, llama.text)

            if conf >= CONFIDENCE_THRESHOLD:
                return RouterResult(
                    answer=llama.text,
                    model_used="llama3-bedrock",
                    complexity=clf.complexity,
                    confidence=conf,
                    escalated=False,
                    attempted=attempted,
                    errors=errors,
                ).to_dict()

        else:
            errors[llama.model] = llama.error or "unknown"

        # escalate to Claude
        claude = _invoke_claude(prompt)

        attempted.append(claude.model)

        if claude.success and claude.text:
            conf = score_confidence(query, claude.text)

            return RouterResult(
                answer=claude.text,
                model_used="claude-sonnet",
                complexity=clf.complexity,
                confidence=conf,
                escalated=True,
                attempted=attempted,
                errors=errors,
            ).to_dict()

        errors[claude.model] = claude.error or "unknown"

    # ─── COMPLEX ──────────────────────────────────────────────────────────────

    # ─── COMPLEX ──────────────────────────────────────────────────────────────

    else:
        claude = _invoke_claude(prompt)

        attempted.append(claude.model)

        if claude.success and claude.text:
            conf = score_confidence(query, claude.text)

            if conf >= CONFIDENCE_THRESHOLD:
                return RouterResult(
                    answer=claude.text,
                    model_used="claude-sonnet",
                    complexity=clf.complexity,
                    confidence=conf,
                    escalated=False,
                    attempted=attempted,
                    errors=errors,
                ).to_dict()

            # retry with more tokens
            retry = _invoke_claude(
                prompt,
                max_tokens=2048,
            )

            attempted.append("claude-sonnet-retry")

            if retry.success and retry.text:
                retry_conf = score_confidence(
                    query,
                    retry.text,
                )

                return RouterResult(
                    answer=retry.text,
                    model_used="claude-sonnet-retry",
                    complexity=clf.complexity,
                    confidence=retry_conf,
                    escalated=True,
                    attempted=attempted,
                    errors=errors,
                ).to_dict()

            errors["claude-sonnet-retry"] = retry.error or "unknown"

        else:
            errors[claude.model] = claude.error or "unknown"

    # ─── Ollama fallback ──────────────────────────────────────────────────────

    ollama = _invoke_ollama(prompt)

    attempted.append(ollama.model)

    if ollama.success and ollama.text:
        conf = score_confidence(query, ollama.text)

        return RouterResult(
            answer=ollama.text,
            model_used=ollama.model,
            complexity=clf.complexity,
            confidence=conf,
            escalated=True,
            attempted=attempted,
            errors=errors,
        ).to_dict()

    errors[ollama.model] = ollama.error or "unknown"

    return RouterResult(
        answer="Error: all model tiers failed to generate a response.",
        model_used="none",
        complexity=clf.complexity,
        confidence=0.0,
        escalated=True,
        attempted=attempted,
        errors=errors,
    ).to_dict()
