import json
from unittest.mock import MagicMock, patch

import pytest

from backend.app.bedrock_router import (
    CONFIDENCE_THRESHOLD,
    ModelResponse,
    classify_complexity,
    route_and_invoke,
    score_confidence,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _llama(text: str) -> ModelResponse:
    return ModelResponse(model="llama3-bedrock", text=text, success=True)


def _claude(text: str) -> ModelResponse:
    return ModelResponse(model="claude-sonnet", text=text, success=True)


def _ollama(text: str) -> ModelResponse:
    return ModelResponse(model="ollama-llama3", text=text, success=True)


def _fail(model: str, error: str = "timeout") -> ModelResponse:
    return ModelResponse(model=model, text="", success=False, error=error)


def _good_answer() -> str:
    return (
        "The answer is confirmed by multiple sources published in 2023. "
        "Key findings include:\n- Point one\n- Point two\n- Point three"
    )


def _weak_answer() -> str:
    return "I'm not sure, possibly correct but you may want to verify this."


# ─── Complexity Classifier ────────────────────────────────────────────────────


class TestClassifyComplexity:
    def test_simple_factual(self):
        assert classify_complexity("who invented the telephone").complexity == "simple"

    def test_complex_reasoning(self):
        r = classify_complexity(
            "analyze the trade-offs between microservices and monoliths"
        )
        assert r.complexity == "complex"

    def test_long_query_bumps_complex(self):
        q = "what are all the different factors that influence the performance of a RAG pipeline"
        assert classify_complexity(q).complexity == "complex"

    def test_long_context_bumps_complex(self):
        assert (
            classify_complexity("summarize this", context="x " * 600).complexity
            == "complex"
        )

    def test_empty_defaults_to_simple(self):
        assert classify_complexity("").complexity == "simple"

    def test_signals_populated(self):
        r = classify_complexity("analyze the implications of this policy change")
        assert len(r.signals) > 0

    def test_complex_keyword_overrides_simple(self):
        r = classify_complexity("what does this analyze mean for strategy")
        assert r.complexity == "complex"

    def test_subordinate_clause_adds_score(self):
        r1 = classify_complexity("list items")
        r2 = classify_complexity("list items because we need to understand why")
        assert r2.score > r1.score


# ─── Confidence Scorer ────────────────────────────────────────────────────────


class TestScoreConfidence:
    def test_empty_returns_zero(self):
        assert score_confidence("q", "") == 0.0
        assert score_confidence("q", "   ") == 0.0

    def test_hedging_lowers_score(self):
        assert score_confidence("q", _weak_answer()) < CONFIDENCE_THRESHOLD

    def test_structured_concrete_scores_high(self):
        answer = (
            "The Python GIL was introduced in 1992 by Guido van Rossum.\n"
            "- It prevents true parallelism in CPython\n"
            "- It simplifies memory management\n"
            "- Removed in Python 3.13 (PEP 703)"
        )
        assert (
            score_confidence("explain the Python GIL", answer) >= CONFIDENCE_THRESHOLD
        )

    def test_very_short_answer_low_confidence(self):
        assert (
            score_confidence("explain quantum entanglement in detail", "Yes.")
            < CONFIDENCE_THRESHOLD
        )

    def test_refusal_phrase_drops_score(self):
        assert (
            score_confidence("q", "No information is available for this query.")
            < CONFIDENCE_THRESHOLD
        )

    def test_score_bounded_zero_to_one(self):
        for ans in ["", "yes", "x " * 200, _weak_answer()]:
            s = score_confidence("query", ans)
            assert 0.0 <= s <= 1.0


# ─── Simple path ──────────────────────────────────────────────────────────────


class TestSimplePath:
    def test_llama_high_confidence_no_claude(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_llama(_good_answer()),
            ),
            patch("backend.app.bedrock_router._invoke_claude") as no_claude,
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("prompt", "who invented radium")

        assert result["model_used"] == "llama3-bedrock"
        assert result["escalated"] is False
        no_claude.assert_not_called()

    def test_llama_low_confidence_escalates_to_claude(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_llama(_weak_answer()),
            ),
            patch(
                "backend.app.bedrock_router._invoke_claude",
                return_value=_claude(_good_answer()),
            ),
            patch("backend.app.bedrock_router._invoke_ollama") as no_ollama,
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("prompt", "who invented radium")

        assert result["model_used"] == "claude-sonnet"
        assert result["escalated"] is True
        assert "llama3-bedrock" in result["attempted"]
        no_ollama.assert_not_called()

    def test_llama_fails_escalates_to_claude(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_fail("llama3-bedrock"),
            ),
            patch(
                "backend.app.bedrock_router._invoke_claude",
                return_value=_claude(_good_answer()),
            ),
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("prompt", "q")

        assert result["model_used"] == "claude-sonnet"
        assert "llama3-bedrock" in result["errors"]


# ─── Complex path ─────────────────────────────────────────────────────────────


class TestComplexPath:
    def test_claude_high_confidence_no_llama(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_claude",
                return_value=_claude(_good_answer()),
            ),
            patch("backend.app.bedrock_router._invoke_llama") as no_llama,
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("complex", 0.8, ["complex keywords"])
            result = route_and_invoke("prompt", "analyze microservices vs monoliths")

        assert result["model_used"] == "claude-sonnet"
        assert result["escalated"] is False
        no_llama.assert_not_called()

    def test_claude_low_confidence_retries_with_more_tokens(self):
        call_count = {"n": 0}

        def claude_side_effect(prompt, max_tokens=1024):
            call_count["n"] += 1
            return (
                _claude(_weak_answer())
                if call_count["n"] == 1
                else _claude(_good_answer())
            )

        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_claude",
                side_effect=claude_side_effect,
            ),
            patch("backend.app.bedrock_router._invoke_ollama") as no_ollama,
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("complex", 0.8, [])
            result = route_and_invoke("prompt", "analyze microservices")

        assert "retry" in result["model_used"]
        assert result["escalated"] is True
        assert call_count["n"] == 2
        no_ollama.assert_not_called()


# ─── Ollama emergency fallback ────────────────────────────────────────────────


class TestOllamaFallback:
    def test_both_bedrock_fail_uses_ollama(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_fail("llama3-bedrock"),
            ),
            patch(
                "backend.app.bedrock_router._invoke_claude",
                return_value=_fail("claude-sonnet"),
            ),
            patch(
                "backend.app.bedrock_router._invoke_ollama",
                return_value=_ollama(_good_answer()),
            ),
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("prompt", "q")

        assert "ollama" in result["model_used"]
        assert result["escalated"] is True
        assert len(result["errors"]) >= 2

    def test_all_fail_returns_error_message(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_fail("llama3-bedrock"),
            ),
            patch(
                "backend.app.bedrock_router._invoke_claude",
                return_value=_fail("claude-sonnet"),
            ),
            patch(
                "backend.app.bedrock_router._invoke_ollama",
                return_value=_fail("ollama-llama3"),
            ),
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("prompt", "q")

        assert result["model_used"] == "none"
        assert "Error" in result["answer"]
        assert result["confidence"] == 0.0


# ─── RouterResult structure ───────────────────────────────────────────────────


class TestRouterResultStructure:
    def test_all_required_keys_present(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_llama(_good_answer()),
            ),
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("p", "q")

        for key in (
            "answer",
            "model_used",
            "complexity",
            "confidence",
            "escalated",
            "attempted",
            "errors",
        ):
            assert key in result, f"missing key: {key}"

    def test_confidence_is_bounded_float(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_llama(_good_answer()),
            ),
        ):
            from backend.app.bedrock_router import ClassifierResult

            mc.return_value = ClassifierResult("simple", 0.1, [])
            result = route_and_invoke("p", "q")

        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
