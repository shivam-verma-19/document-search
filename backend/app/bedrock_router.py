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

# ─── Complexity classifier signals ────────────────────────────────────────────

_COMPLEX_KEYWORDS = {
    "analyze",
    "analyse",
    "compare",
    "contrast",
    "evaluate",
    "assess",
    "explain why",
    "what is the impact",
    "implications",
    "trade-off",
    "pros and cons",
    "advantages and disadvantages",
    "summarize",
    "summarise",
    "synthesize",
    "how does",
    "reasoning",
    "argue",
    "critique",
    "recommend",
    "strategy",
    "relationship between",
}

_SIMPLE_KEYWORDS = {
    "who",
    "when",
    "where",
    "what is",
    "what are",
    "define",
    "list",
    "how many",
    "how much",
    "give me",
    "show me",
    "tell me",
    "find",
    "name",
    "which year",
    "what date",
}

_HEDGING_PHRASES = [
    r"\bi (am not sure|don'?t know|cannot be certain|may be wrong)\b",
    r"\b(it (is|might be) possible|possibly|perhaps|might be|could be)\b",
    r"\b(i (think|believe|assume)|as far as i know|to my knowledge)\b",
    r"\b(unclear|uncertain|not (entirely |completely )?sure|hard to say)\b",
    r"\b(you (may|might|should) (want to |)verify|please (check|confirm|verify))\b",
    r"\b(i (don'?t|do not) have (enough |)(information|context|data))\b",
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

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "model_used": self.model_used,
            "complexity": self.complexity,
            "confidence": round(self.confidence, 3),
            "escalated": self.escalated,
            "attempted": self.attempted,
            "errors": self.errors,
        }


# ─── Bedrock client (lazy singleton) ─────────────────────────────────────────

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


# ─── Step 1: Complexity Classifier ───────────────────────────────────────────


def classify_complexity(query: str, context: str = "") -> ClassifierResult:
    lower = query.lower()
    signals = []
    score = 0.0

    matched_complex = [kw for kw in _COMPLEX_KEYWORDS if kw in lower]
    if matched_complex:
        score += 0.30
        signals.append(f"complex keywords: {matched_complex[:2]}")

    matched_simple = [kw for kw in _SIMPLE_KEYWORDS if kw in lower]
    if matched_simple and not matched_complex:
        score -= 0.25
        signals.append(f"simple keywords: {matched_simple[:2]}")

    token_count = len(query.split())
    if token_count > 15:
        score += 0.20
        signals.append(f"long query ({token_count} tokens)")

    sentence_count = len(re.split(r"[.!?]+", query.strip()))
    if sentence_count > 1:
        score += 0.15
        signals.append(f"multi-sentence ({sentence_count})")

    if re.search(r"\b(why|how|because|since|although|whereas|which means)\b", lower):
        score += 0.15
        signals.append("subordinate clause")

    if len(context) > 1000:
        score += 0.20
        signals.append(f"long context ({len(context)} chars)")

    complexity: QueryComplexity = "complex" if score >= 0.40 else "simple"
    logger.info("Classifier: %s (score=%.2f) signals=%s", complexity, score, signals)
    return ClassifierResult(complexity=complexity, score=score, signals=signals)


# ─── Step 3: Confidence Scorer ───────────────────────────────────────────────


def score_confidence(query: str, answer: str) -> float:
    if not answer or not answer.strip():
        return 0.0

    score = 0.75
    answer_words = len(answer.split())
    query_words = len(query.split())

    hedge_hits = sum(1 for p in _HEDGING_RE if p.search(answer))
    score -= hedge_hits * 0.12

    if answer_words < 10:
        score -= 0.40
    elif answer_words < 20:
        score -= 0.20

    if query_words > 10 and answer_words < query_words:
        score -= 0.10

    if re.search(r"(\n[-*•]|\n\d+\.)", answer):
        score += 0.10

    concrete = len(
        re.findall(r"\b(\d{4}|\d+%|\$[\d,]+|[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b", answer)
    )
    score += min(concrete * 0.05, 0.15)

    if re.search(
        r"(no information|not (found|available|provided)|cannot answer)", answer, re.I
    ):
        score -= 0.30

    return max(0.0, min(1.0, score))


# ─── Step 2: Model invokers ───────────────────────────────────────────────────


def _invoke_claude(prompt: str, max_tokens: int = 1024) -> ModelResponse:
    """Primary — Claude Sonnet via Bedrock Messages API."""
    try:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        resp = _get_bedrock_client().invoke_model(
            modelId=CLAUDE_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        text = json.loads(resp["body"].read())["content"][0]["text"].strip()
        logger.info("Claude Sonnet responded (len=%d)", len(text))
        return ModelResponse(model="claude-sonnet", text=text, success=True)
    except Exception as exc:
        logger.warning("Claude Sonnet failed: %s", exc)
        return ModelResponse(
            model="claude-sonnet", text="", success=False, error=str(exc)
        )


def _invoke_llama(prompt: str) -> ModelResponse:
    """Secondary — Llama 3 via Bedrock."""
    try:
        body = json.dumps(
            {
                "prompt": (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                    f"{prompt}"
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                ),
                "max_gen_len": 1024,
                "temperature": 0.3,
                "top_p": 0.9,
            }
        )
        resp = _get_bedrock_client().invoke_model(
            modelId=LLAMA_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        text = json.loads(resp["body"].read()).get("generation", "").strip()
        logger.info("Llama 3 responded (len=%d)", len(text))
        return ModelResponse(model="llama3-bedrock", text=text, success=True)
    except Exception as exc:
        logger.warning("Llama 3 failed: %s", exc)
        return ModelResponse(
            model="llama3-bedrock", text="", success=False, error=str(exc)
        )


def _invoke_ollama(prompt: str) -> ModelResponse:
    """Emergency local fallback — only reached if both Bedrock tiers fail."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if not text:
            raise ValueError("Empty response from Ollama")
        logger.info("Ollama (%s) responded (len=%d)", OLLAMA_MODEL, len(text))
        return ModelResponse(model=f"ollama-{OLLAMA_MODEL}", text=text, success=True)
    except Exception as exc:
        logger.error("Ollama fallback failed: %s", exc)
        return ModelResponse(
            model=f"ollama-{OLLAMA_MODEL}", text="", success=False, error=str(exc)
        )


# ─── Main router ──────────────────────────────────────────────────────────────


def route_and_invoke(prompt: str, query: str = "", context: str = "") -> dict:
    attempted: list[str] = []
    errors: dict[str, str] = {}
    escalated = False

    clf = classify_complexity(query, context)

    # ── Simple path: Llama first ───────────────────────────────────────────────
    if clf.complexity == "simple":
        primary = _invoke_llama(prompt)
        attempted.append(primary.model)

        if primary.success and primary.text:
            conf = score_confidence(query, primary.text)
            logger.info(
                "Llama confidence=%.2f threshold=%.2f", conf, CONFIDENCE_THRESHOLD
            )

            if conf >= CONFIDENCE_THRESHOLD:
                return RouterResult(
                    answer=primary.text,
                    model_used=primary.model,
                    complexity=clf.complexity,
                    confidence=conf,
                    escalated=False,
                    attempted=attempted,
                    errors=errors,
                ).to_dict()

            logger.info("Low Llama confidence — escalating to Claude")
            escalated = True
        else:
            errors[primary.model] = primary.error or "unknown"
            escalated = True

        # Escalate to Claude
        secondary = _invoke_claude(prompt)
        attempted.append(secondary.model)

        if secondary.success and secondary.text:
            conf = score_confidence(query, secondary.text)
            if conf >= CONFIDENCE_THRESHOLD or not errors:
                return RouterResult(
                    answer=secondary.text,
                    model_used=secondary.model,
                    complexity=clf.complexity,
                    confidence=conf,
                    escalated=escalated,
                    attempted=attempted,
                    errors=errors,
                ).to_dict()
            errors[secondary.model] = f"low confidence ({conf:.2f})"
        else:
            errors[secondary.model] = secondary.error or "unknown"

    # ── Complex path: Claude first ────────────────────────────────────────────
    else:
        primary = _invoke_claude(prompt)
        attempted.append(primary.model)

        if primary.success and primary.text:
            conf = score_confidence(query, primary.text)
            logger.info(
                "Claude confidence=%.2f threshold=%.2f", conf, CONFIDENCE_THRESHOLD
            )

            if conf >= CONFIDENCE_THRESHOLD:
                return RouterResult(
                    answer=primary.text,
                    model_used=primary.model,
                    complexity=clf.complexity,
                    confidence=conf,
                    escalated=False,
                    attempted=attempted,
                    errors=errors,
                ).to_dict()

            # Retry Claude with expanded tokens
            logger.info("Low Claude confidence — retrying with 2048 tokens")
            escalated = True
            retry = _invoke_claude(prompt, max_tokens=2048)
            attempted.append(f"{retry.model}(retry)")

            if retry.success and retry.text:
                retry_conf = score_confidence(query, retry.text)
                if retry_conf >= CONFIDENCE_THRESHOLD:
                    return RouterResult(
                        answer=retry.text,
                        model_used=f"{retry.model}(retry)",
                        complexity=clf.complexity,
                        confidence=retry_conf,
                        escalated=True,
                        attempted=attempted,
                        errors=errors,
                    ).to_dict()
                errors[f"{retry.model}(retry)"] = f"low confidence ({retry_conf:.2f})"
            else:
                errors[f"{retry.model}(retry)"] = retry.error or "unknown"
        else:
            errors[primary.model] = primary.error or "unknown"
            escalated = True

    # ── Emergency: Ollama ─────────────────────────────────────────────────────
    logger.warning("Both Bedrock tiers exhausted — Ollama emergency fallback")
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

    logger.error("All models failed. errors=%s", errors)
    return RouterResult(
        answer="Error: all model tiers failed to generate a response.",
        model_used="none",
        complexity=clf.complexity,
        confidence=0.0,
        escalated=True,
        attempted=attempted,
        errors=errors,
    ).to_dict()
