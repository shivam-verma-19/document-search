"""
Tests for gemini_client.py — single model (gemini-2.5-flash, free tier).
All Gemini API calls are mocked. Runs fully offline.
"""

import sys
from unittest.mock import patch

for _key in [k for k in list(sys.modules) if "gemini_client" in k and "test" not in k]:
    del sys.modules[_key]

import backend.app.gemini_client as _client  # noqa: E402

MODEL = _client.GEMINI_MODEL
MODEL_RETRY = f"{MODEL}-retry"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ok(text: str) -> "_client.ModelResponse":
    return _client.ModelResponse(model=MODEL, text=text, success=True)


def _fail(error: str = "timeout") -> "_client.ModelResponse":
    return _client.ModelResponse(model=MODEL, text="", success=False, error=error)


def _good_answer() -> str:
    return (
        "The answer is confirmed by multiple sources published in 2023.\n"
        "- Point one\n- Point two\n- Point three"
    )


def _weak_answer() -> str:
    return "I'm not sure, possibly correct but verify this."


# ── Complexity ────────────────────────────────────────────────────────────────


class TestClassifyComplexity:
    def test_simple_factual(self):
        assert (
            _client.classify_complexity("who invented the telephone").complexity
            == "simple"
        )

    def test_complex_reasoning(self):
        q = "Analyze the trade-offs between microservices and monoliths and explain why scalability differs."
        assert _client.classify_complexity(q).complexity == "complex"

    def test_long_context_bumps_complex(self):
        r = _client.classify_complexity("summarize this", context="x " * 600)
        assert r.complexity == "complex"

    def test_empty_defaults_to_simple(self):
        assert _client.classify_complexity("").complexity == "simple"

    def test_signals_populated(self):
        r = _client.classify_complexity(
            "analyze the implications of this policy change"
        )
        assert len(r.signals) > 0

    def test_subordinate_clause_adds_score(self):
        r1 = _client.classify_complexity("list items")
        r2 = _client.classify_complexity("list items because we need to understand why")
        assert r2.score > r1.score


# ── Confidence ────────────────────────────────────────────────────────────────


class TestScoreConfidence:
    def test_empty_returns_zero(self):
        assert _client.score_confidence("q", "") == 0.0

    def test_hedging_lowers_score(self):
        assert (
            _client.score_confidence("q", _weak_answer()) < _client.CONFIDENCE_THRESHOLD
        )

    def test_good_answer_above_threshold(self):
        assert (
            _client.score_confidence("q", _good_answer())
            >= _client.CONFIDENCE_THRESHOLD
        )

    def test_very_short_low_confidence(self):
        assert (
            _client.score_confidence("explain quantum entanglement", "Yes.")
            < _client.CONFIDENCE_THRESHOLD
        )

    def test_score_bounded(self):
        for ans in ["", "yes", "x " * 200, _weak_answer()]:
            s = _client.score_confidence("query", ans)
            assert 0.0 <= s <= 1.0


# ── Router: high confidence → no retry ───────────────────────────────────────


class TestRouterHighConfidence:
    def test_good_answer_returned_directly(self):
        with patch.object(
            _client, "_invoke_gemini", return_value=_ok(_good_answer())
        ) as mock_invoke:
            result = _client.route_and_invoke("prompt", "who invented radium")

        assert result["answer"] != ""
        assert result["escalated"] is False
        assert result["model_used"] == MODEL
        assert mock_invoke.call_count == 1

    def test_simple_query_uses_1024_tokens(self):
        with patch.object(_client, "classify_complexity") as mc, patch.object(
            _client, "_invoke_gemini", return_value=_ok(_good_answer())
        ) as mock_invoke:
            mc.return_value = _client.ClassifierResult("simple", 0.1, [])
            _client.route_and_invoke("prompt", "who is einstein")

        mock_invoke.assert_called_once_with("prompt", max_tokens=1024)

    def test_complex_query_uses_2048_tokens(self):
        with patch.object(_client, "classify_complexity") as mc, patch.object(
            _client, "_invoke_gemini", return_value=_ok(_good_answer())
        ) as mock_invoke:
            mc.return_value = _client.ClassifierResult(
                "complex", 0.6, ["complex keywords"]
            )
            _client.route_and_invoke("prompt", "analyze the implications")

        mock_invoke.assert_called_once_with("prompt", max_tokens=2048)


# ── Router: low confidence → retry ───────────────────────────────────────────


class TestRouterRetry:
    def test_low_confidence_triggers_retry(self):
        call_count = {"n": 0}

        def fake_invoke(prompt, max_tokens=1024):
            call_count["n"] += 1
            return _ok(_weak_answer()) if call_count["n"] == 1 else _ok(_good_answer())

        with patch.object(_client, "_invoke_gemini", side_effect=fake_invoke):
            result = _client.route_and_invoke("prompt", "explain something")

        assert call_count["n"] == 2
        assert result["escalated"] is True
        assert MODEL_RETRY in result["attempted"]

    def test_simple_retry_uses_2048_tokens(self):
        calls = []

        def fake_invoke(prompt, max_tokens=1024):
            calls.append(max_tokens)
            return _ok(_weak_answer()) if len(calls) == 1 else _ok(_good_answer())

        with patch.object(_client, "classify_complexity") as mc, patch.object(
            _client, "_invoke_gemini", side_effect=fake_invoke
        ):
            mc.return_value = _client.ClassifierResult("simple", 0.1, [])
            _client.route_and_invoke("prompt", "q")

        assert calls == [1024, 2048]

    def test_complex_retry_uses_4096_tokens(self):
        calls = []

        def fake_invoke(prompt, max_tokens=2048):
            calls.append(max_tokens)
            return _ok(_weak_answer()) if len(calls) == 1 else _ok(_good_answer())

        with patch.object(_client, "classify_complexity") as mc, patch.object(
            _client, "_invoke_gemini", side_effect=fake_invoke
        ):
            mc.return_value = _client.ClassifierResult("complex", 0.6, [])
            _client.route_and_invoke("prompt", "analyze this")

        assert calls == [2048, 4096]


# ── Router: total failure ─────────────────────────────────────────────────────


class TestRouterFailure:
    def test_both_fail_returns_error(self):
        with patch.object(_client, "_invoke_gemini", return_value=_fail("api error")):
            result = _client.route_and_invoke("prompt", "q")

        assert "Error" in result["answer"]
        assert result["model_used"] == "none"
        assert result["confidence"] == 0.0
