"""Tests for eval.py — retrieval quality (recall@k, MRR) and answer quality scoring."""

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _evict_stubs():
    """Remove any MagicMock stubs installed by test_rag so real modules load."""
    for mod in ["backend.app.eval"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.eval"]:
        sys.modules.pop(mod, None)


from unittest.mock import MagicMock, patch

import pytest


def _make_doc(doc_id: str, text: str = "content"):
    from backend.app.document_repository import SearchDocument

    return SearchDocument(page_content=text, doc_id=doc_id)


class TestEvaluateRetrieval:
    def test_perfect_recall(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("id1"), _make_doc("id2"), _make_doc("id3")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=["id1", "id2", "id3"])
        assert metrics.recall_at_k == pytest.approx(1.0)

    def test_zero_recall(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("id1"), _make_doc("id2")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=["id99"])
        assert metrics.recall_at_k == pytest.approx(0.0)

    def test_partial_recall(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("id1"), _make_doc("id2"), _make_doc("id3")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=["id1", "id99"])
        assert metrics.recall_at_k == pytest.approx(0.5)

    def test_mrr_first_hit_at_rank_1(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("relevant"), _make_doc("other")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=["relevant"])
        assert metrics.mrr == pytest.approx(1.0)

    def test_mrr_first_hit_at_rank_2(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("other"), _make_doc("relevant")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=["relevant"])
        assert metrics.mrr == pytest.approx(0.5)

    def test_mrr_no_relevant_docs(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("id1"), _make_doc("id2")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=["id99"])
        assert metrics.mrr == pytest.approx(0.0)

    def test_no_relevant_doc_ids_returns_zero_scores(self):
        from backend.app.eval import evaluate_retrieval

        docs = [_make_doc("id1")]
        metrics = evaluate_retrieval("q", docs, relevant_doc_ids=None)
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.retrieved_count == 1

    def test_empty_retrieved_docs(self):
        from backend.app.eval import evaluate_retrieval

        metrics = evaluate_retrieval("q", [], relevant_doc_ids=["id1"])
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0


class TestEvaluateAnswer:
    def test_returns_answer_metrics_object(self):
        from backend.app.eval import evaluate_answer

        with patch("backend.app.eval._llm_score", return_value=0.9):
            result = evaluate_answer("question", "context here", "answer text")
        assert hasattr(result, "faithfulness")
        assert hasattr(result, "relevance")
        assert hasattr(result, "llm_judged")

    def test_llm_judged_true_when_enabled(self, monkeypatch):
        import backend.app.eval as ev

        monkeypatch.setattr(ev, "EVAL_LLM_ENABLED", True)

        with patch.object(ev, "_llm_score", return_value=0.8):
            result = ev.evaluate_answer("q", "context", "answer")
        assert result.llm_judged is True

    def test_llm_judged_false_when_disabled(self, monkeypatch):
        import backend.app.eval as ev

        monkeypatch.setattr(ev, "EVAL_LLM_ENABLED", False)

        result = ev.evaluate_answer("q", "context", "answer")
        assert result.llm_judged is False
        assert result.faithfulness == 0.5  # placeholder
        assert result.relevance == 0.5

    def test_empty_answer_returns_zeros(self, monkeypatch):
        import backend.app.eval as ev

        monkeypatch.setattr(ev, "EVAL_LLM_ENABLED", True)

        result = ev.evaluate_answer("q", "ctx", "")
        assert result.faithfulness == 0.0
        assert result.relevance == 0.0

    def test_scores_bounded_0_to_1(self):
        from backend.app.eval import evaluate_answer

        with patch("backend.app.eval._llm_score", return_value=0.75):
            result = evaluate_answer("q", "ctx", "answer text here")
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.relevance <= 1.0


class TestLogEval:
    def test_log_eval_does_not_raise_on_dynamo_failure(self):
        from backend.app.eval import AnswerMetrics, RetrievalMetrics, log_eval

        with patch("backend.app.eval._get_table") as mock_table:
            mock_table.return_value.put_item.side_effect = Exception("dynamo down")
            # Must not raise
            log_eval("query", RetrievalMetrics(), AnswerMetrics())

    def test_log_eval_writes_to_dynamo(self):
        from backend.app.eval import AnswerMetrics, RetrievalMetrics, log_eval

        with patch("backend.app.eval._get_table") as mock_table, patch(
            "backend.app.eval.push_metric"
        ):
            log_eval(
                "test query",
                RetrievalMetrics(
                    recall_at_k=0.8, mrr=1.0, retrieved_count=3, relevant_count=3
                ),
                AnswerMetrics(faithfulness=0.9, relevance=0.85, llm_judged=True),
            )
            mock_table.return_value.put_item.assert_called_once()
            item = mock_table.return_value.put_item.call_args.kwargs["Item"]
            assert item["query"] == "test query"
            assert "faithfulness" in item
            assert "recall_at_k" in item

    def test_cloudwatch_metrics_pushed(self):
        from backend.app.eval import AnswerMetrics, RetrievalMetrics, log_eval

        with patch("backend.app.eval._get_table") as mock_table, patch(
            "backend.app.eval.push_metric"
        ) as mock_push:
            mock_table.return_value.put_item.return_value = {}
            log_eval(
                "q",
                RetrievalMetrics(
                    recall_at_k=0.5, mrr=0.5, retrieved_count=2, relevant_count=2
                ),
                AnswerMetrics(faithfulness=0.8, relevance=0.7, llm_judged=True),
            )
            metric_names = [c.args[0] for c in mock_push.call_args_list]
            assert "EvalFaithfulness" in metric_names
            assert "EvalRelevance" in metric_names


class TestLLMScore:
    @pytest.fixture(autouse=True)
    def _reset_gemini_client(self):
        """Reset cached _client so per-test patches on _get_client take effect."""
        import backend.app.gemini_client as gc

        gc._client = None
        yield
        gc._client = None

    def test_parses_valid_float(self):
        from backend.app.eval import _llm_score

        mock_response = MagicMock()
        mock_response.text = "0.85"
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value.text = "0.85"
            score = _llm_score("some prompt")
        assert score == pytest.approx(0.85)

    def test_clamps_to_0_1(self):
        from backend.app.eval import _llm_score

        mock_response = MagicMock()
        mock_response.text = "1.5"  # out of range
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value.text = "0.85"
            score = _llm_score("prompt")
        assert score == pytest.approx(1.0)

    def test_returns_0_5_on_parse_failure(self):
        from backend.app.eval import _llm_score

        mock_response = MagicMock()
        mock_response.text = "not a number"
        with patch("backend.app.gemini_client._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value.text = "0.85"
            score = _llm_score("prompt")
        assert score == pytest.approx(0.5)

    def test_returns_0_5_on_exception(self):
        from backend.app.eval import _llm_score

        with patch(
            "backend.app.gemini_client._get_client", side_effect=Exception("down")
        ):
            score = _llm_score("prompt")
        assert score == pytest.approx(0.5)
