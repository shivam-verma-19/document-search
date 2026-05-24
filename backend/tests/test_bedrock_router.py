import importlib
import sys
import unittest.mock as mock
from unittest.mock import patch

# ── Force-reload the router so stale .pyc bytecode can never interfere ────────
# Evict from sys.modules first, then re-import fresh from source.
for _key in [k for k in list(sys.modules) if "bedrock_router" in k and "test" not in k]:
    del sys.modules[_key]

import backend.app.bedrock_router as _router  # noqa: E402  (must come after eviction)

# All references go through _router.* — never via direct `from ... import`
# so patches and reloads always affect the same object the tests call.

# ── Model name constants ───────────────────────────────────────────────────────
# Update these if you change models; every assertion below uses them.
LLAMA_MODEL = "llama3-bedrock"
CLAUDE_MODEL = "claude-sonnet-4-5"
CLAUDE_RETRY = "claude-sonnet-4-5-retry"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _llama(text: str) -> "_router.ModelResponse":
    return _router.ModelResponse(model=LLAMA_MODEL, text=text, success=True)


def _claude(text: str) -> "_router.ModelResponse":
    return _router.ModelResponse(model=CLAUDE_MODEL, text=text, success=True)


def _fail(model: str, error: str = "timeout") -> "_router.ModelResponse":
    return _router.ModelResponse(model=model, text="", success=False, error=error)


def _good_answer() -> str:
    return (
        "The answer is confirmed by multiple sources published in 2023.\n"
        "- Point one\n"
        "- Point two\n"
        "- Point three"
    )


def _weak_answer() -> str:
    return "I'm not sure, possibly correct but verify this."


# ── Complexity ────────────────────────────────────────────────────────────────


class TestClassifyComplexity:
    def test_simple_factual(self):
        r = _router.classify_complexity("who invented the telephone")
        assert r.complexity == "simple"

    def test_complex_reasoning(self):
        q = (
            "Analyze the trade-offs between microservices and monoliths "
            "and explain why scalability differs."
        )
        assert _router.classify_complexity(q).complexity == "complex"

    def test_long_query_bumps_score(self):
        q = (
            "What are all the different architectural and retrieval factors "
            "that influence the performance of a hybrid RAG pipeline "
            "in production systems?"
        )
        assert _router.classify_complexity(q).complexity in ("simple", "complex")

    def test_long_context_bumps_complex(self):
        r = _router.classify_complexity("summarize this", context="x " * 600)
        assert r.complexity == "complex"

    def test_empty_defaults_to_simple(self):
        assert _router.classify_complexity("").complexity == "simple"

    def test_signals_populated(self):
        r = _router.classify_complexity(
            "analyze the implications of this policy change"
        )
        assert len(r.signals) > 0

    def test_complex_keyword_overrides_simple(self):
        r = _router.classify_complexity(
            "what does this analyze mean for strategy and scalability?"
        )
        assert r.complexity == "complex"

    def test_subordinate_clause_adds_score(self):
        r1 = _router.classify_complexity("list items")
        r2 = _router.classify_complexity("list items because we need to understand why")
        assert r2.score > r1.score


# ── Confidence ────────────────────────────────────────────────────────────────


class TestScoreConfidence:
    def test_empty_returns_zero(self):
        assert _router.score_confidence("q", "") == 0.0
        assert _router.score_confidence("q", "   ") == 0.0

    def test_hedging_lowers_score(self):
        assert (
            _router.score_confidence("q", _weak_answer()) < _router.CONFIDENCE_THRESHOLD
        )

    def test_structured_concrete_scores_high(self):
        answer = (
            "The Python GIL was introduced in 1992.\n"
            "- Prevents true parallelism\n"
            "- Simplifies memory management\n"
            "- Related to CPython internals"
        )
        assert (
            _router.score_confidence("explain GIL", answer)
            >= _router.CONFIDENCE_THRESHOLD
        )

    def test_very_short_answer_low_confidence(self):
        assert (
            _router.score_confidence("explain quantum entanglement", "Yes.")
            < _router.CONFIDENCE_THRESHOLD
        )

    def test_refusal_phrase_drops_score(self):
        assert (
            _router.score_confidence("q", "No information is available for this query.")
            < _router.CONFIDENCE_THRESHOLD
        )

    def test_score_bounded_zero_to_one(self):
        for ans in ["", "yes", "x " * 200, _weak_answer()]:
            s = _router.score_confidence("query", ans)
            assert 0.0 <= s <= 1.0

    def test_good_answer_is_above_threshold(self):
        assert (
            _router.score_confidence("q", _good_answer())
            >= _router.CONFIDENCE_THRESHOLD
        )

    def test_weak_answer_is_below_threshold(self):
        assert (
            _router.score_confidence("q", _weak_answer()) < _router.CONFIDENCE_THRESHOLD
        )


# ── Simple Path ───────────────────────────────────────────────────────────────


class TestSimplePath:
    def test_llama_high_confidence_returns_directly(self):
        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(_router, "_invoke_llama", return_value=_llama(_good_answer())),
            patch.object(_router, "_invoke_claude") as mock_claude,
        ):
            mc.return_value = _router.ClassifierResult("simple", 0.1, [])
            result = _router.route_and_invoke("prompt", "who invented radium")

        assert result["answer"] != ""
        assert result["escalated"] is False
        assert result["model_used"] == LLAMA_MODEL
        mock_claude.assert_not_called()

    def test_llama_low_confidence_escalates_to_claude(self):
        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(_router, "_invoke_llama", return_value=_llama(_weak_answer())),
            patch.object(
                _router, "_invoke_claude", return_value=_claude(_good_answer())
            ),
        ):
            mc.return_value = _router.ClassifierResult("simple", 0.1, [])
            result = _router.route_and_invoke("prompt", "who invented radium")

        assert result["answer"] != ""
        assert result["escalated"] is True
        assert LLAMA_MODEL in result["attempted"]
        assert CLAUDE_MODEL in result["attempted"]

    def test_llama_fails_escalates_to_claude(self):
        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(_router, "_invoke_llama", return_value=_fail(LLAMA_MODEL)),
            patch.object(
                _router, "_invoke_claude", return_value=_claude(_good_answer())
            ),
        ):
            mc.return_value = _router.ClassifierResult("simple", 0.1, [])
            result = _router.route_and_invoke("prompt", "q")

        assert result["answer"] != ""
        assert result["escalated"] is True
        assert LLAMA_MODEL in result["errors"]

    def test_both_fail_returns_error_string(self):
        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(_router, "_invoke_llama", return_value=_fail(LLAMA_MODEL)),
            patch.object(_router, "_invoke_claude", return_value=_fail(CLAUDE_MODEL)),
        ):
            mc.return_value = _router.ClassifierResult("simple", 0.1, [])
            result = _router.route_and_invoke("prompt", "q")

        assert "Error" in result["answer"]
        assert result["model_used"] == "none"


# ── Complex Path ──────────────────────────────────────────────────────────────


class TestComplexPath:
    def test_claude_high_confidence_returns_directly(self):
        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(
                _router, "_invoke_claude", return_value=_claude(_good_answer())
            ),
        ):
            mc.return_value = _router.ClassifierResult(
                "complex", 0.6, ["complex keywords"]
            )
            result = _router.route_and_invoke("prompt", "analyze the implications")

        assert result["escalated"] is False
        assert result["model_used"] == CLAUDE_MODEL

    def test_claude_low_confidence_retries(self):
        call_count = {"n": 0}

        def fake_claude(prompt, max_tokens=1024):
            call_count["n"] += 1
            return (
                _claude(_weak_answer())
                if call_count["n"] == 1
                else _claude(_good_answer())
            )

        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(_router, "_invoke_claude", side_effect=fake_claude),
        ):
            mc.return_value = _router.ClassifierResult(
                "complex", 0.6, ["complex keywords"]
            )
            result = _router.route_and_invoke("prompt", "analyze this deeply")

        assert call_count["n"] == 2
        assert result["escalated"] is True
        assert CLAUDE_RETRY in result["attempted"]

    def test_claude_fails_returns_error_string(self):
        with (
            patch.object(_router, "classify_complexity") as mc,
            patch.object(_router, "_invoke_claude", return_value=_fail(CLAUDE_MODEL)),
        ):
            mc.return_value = _router.ClassifierResult("complex", 0.6, [])
            result = _router.route_and_invoke("prompt", "complex query")

        assert "Error" in result["answer"]
        assert result["model_used"] == "none"
