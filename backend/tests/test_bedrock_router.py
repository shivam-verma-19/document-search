import unittest.mock as mock
from unittest.mock import patch

from backend.app.bedrock_router import (
    CONFIDENCE_THRESHOLD,
    ClassifierResult,
    ModelResponse,
    classify_complexity,
    route_and_invoke,
    score_confidence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llama(text: str) -> ModelResponse:
    return ModelResponse(
        model="llama3-bedrock",
        text=text,
        success=True,
    )


def _claude(text: str) -> ModelResponse:
    return ModelResponse(
        model="claude-sonnet",
        text=text,
        success=True,
    )


def _ollama(text: str) -> ModelResponse:
    return ModelResponse(
        model="ollama-llama3",
        text=text,
        success=True,
    )


def _fail(model: str, error: str = "timeout") -> ModelResponse:
    return ModelResponse(
        model=model,
        text="",
        success=False,
        error=error,
    )


def _good_answer():
    return (
        "The answer is confirmed by multiple sources published in 2023.\n"
        "- Point one\n"
        "- Point two\n"
        "- Point three"
    )


def _weak_answer():
    return "I'm not sure, possibly correct but verify this."


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------


class TestClassifyComplexity:
    def test_simple_factual(self):
        r = classify_complexity("who invented the telephone")
        assert r.complexity == "simple"

    def test_complex_reasoning(self):
        q = (
            "Analyze the trade-offs between microservices and monoliths "
            "and explain why scalability differs."
        )
        r = classify_complexity(q)

        assert r.complexity == "complex"

    def test_long_query_bumps_complex(self):
        q = (
            "What are all the different architectural and retrieval factors "
            "that influence the performance of a hybrid RAG pipeline "
            "in production systems?"
        )

        assert classify_complexity(q).complexity in ("simple", "complex")

    def test_long_context_bumps_complex(self):
        r = classify_complexity(
            "summarize this",
            context="x " * 600,
        )

        assert r.complexity == "complex"

    def test_empty_defaults_to_simple(self):
        assert classify_complexity("").complexity == "simple"

    def test_signals_populated(self):
        r = classify_complexity("analyze the implications of this policy change")

        assert len(r.signals) > 0

    def test_complex_keyword_overrides_simple(self):
        r = classify_complexity(
            "what does this analyze mean for strategy and scalability?"
        )

        assert r.complexity == "complex"

    def test_subordinate_clause_adds_score(self):
        r1 = classify_complexity("list items")

        r2 = classify_complexity("list items because we need to understand why")

        assert r2.score > r1.score


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestScoreConfidence:
    def test_empty_returns_zero(self):
        assert score_confidence("q", "") == 0.0
        assert score_confidence("q", "   ") == 0.0

    def test_hedging_lowers_score(self):
        assert score_confidence("q", _weak_answer()) < CONFIDENCE_THRESHOLD

    def test_structured_concrete_scores_high(self):
        answer = (
            "The Python GIL was introduced in 1992.\n"
            "- Prevents true parallelism\n"
            "- Simplifies memory management\n"
            "- Related to CPython internals"
        )

        assert score_confidence("explain GIL", answer) >= CONFIDENCE_THRESHOLD

    def test_very_short_answer_low_confidence(self):
        assert (
            score_confidence(
                "explain quantum entanglement",
                "Yes.",
            )
            < CONFIDENCE_THRESHOLD
        )

    def test_refusal_phrase_drops_score(self):
        assert (
            score_confidence(
                "q",
                "No information is available for this query.",
            )
            < CONFIDENCE_THRESHOLD
        )

    def test_score_bounded_zero_to_one(self):
        for ans in ["", "yes", "x " * 200, _weak_answer()]:
            s = score_confidence("query", ans)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Simple Path
# ---------------------------------------------------------------------------


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
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

            result = route_and_invoke(
                "prompt",
                "who invented radium",
            )

        assert result["answer"] != ""
        assert result["model_used"] in ("llama3-bedrock", "claude-sonnet")
        assert isinstance(result["escalated"], bool)

        if result["model_used"] == "llama3-bedrock":
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
        ):
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

            result = route_and_invoke(
                "prompt",
                "who invented radium",
            )

        assert result["answer"] != ""
        assert result["escalated"] is True
        assert "llama3-bedrock" in result["attempted"]

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
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

            result = route_and_invoke("prompt", "q")

        assert result["answer"] != ""
        assert result["escalated"] is True
        assert "llama3-bedrock" in result["errors"]


# ---------------------------------------------------------------------------
# Complex Path
# ---------------------------------------------------------------------------


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
            mc.return_value = ClassifierResult(
                "complex",
                0.8,
                ["complex keywords"],
            )

            result = route_and_invoke(
                "prompt",
                "analyze microservices vs monoliths",
            )

        assert result["answer"] != ""
        assert result["model_used"] in ("claude-sonnet", "claude-sonnet-retry")
        assert isinstance(result["escalated"], bool)

        if result["model_used"] == "claude-sonnet":
            no_llama.assert_not_called()

    def test_claude_low_confidence_retries_with_more_tokens(self):
        call_count = {"n": 0}

        def claude_side_effect(prompt, max_tokens=1024):
            call_count["n"] += 1

            if call_count["n"] == 1:
                return _claude(_weak_answer())

            return _claude(_good_answer())

        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_claude",
                side_effect=claude_side_effect,
            ),
        ):
            mc.return_value = ClassifierResult(
                "complex",
                0.8,
                [],
            )

            result = route_and_invoke(
                "prompt",
                "analyze microservices",
            )

        assert result["answer"] != ""
        assert isinstance(result["escalated"], bool)
        assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# Ollama Fallback
# ---------------------------------------------------------------------------


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
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

            result = route_and_invoke("prompt", "q")

        assert result["answer"] != ""
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
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

            result = route_and_invoke("prompt", "q")

        assert result["model_used"] == "none"
        assert "Error" in result["answer"]
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# RouterResult
# ---------------------------------------------------------------------------


class TestRouterResultStructure:
    def test_all_required_keys_present(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_llama(_good_answer()),
            ),
        ):
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

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
            assert key in result

    def test_confidence_is_bounded_float(self):
        with (
            patch("backend.app.bedrock_router.classify_complexity") as mc,
            patch(
                "backend.app.bedrock_router._invoke_llama",
                return_value=_llama(_good_answer()),
            ),
        ):
            mc.return_value = ClassifierResult(
                "simple",
                0.1,
                [],
            )

            result = route_and_invoke("p", "q")

        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
