import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock google.genai before any import so embeddings module loads cleanly
_mock_genai = MagicMock()
_mock_genai_client_instance = MagicMock()
_mock_genai.Client.return_value = _mock_genai_client_instance
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", _mock_genai)


class TestMetrics:
    def test_log_metrics_calls_dynamo_and_cloudwatch(self):
        import backend.app.metrics as metrics

        with patch.object(metrics, "_get_table") as mock_get_table, patch(
            "backend.app.metrics.push_metric"
        ) as mock_push:
            mock_table = MagicMock()
            mock_get_table.return_value = mock_table
            metrics.log_metrics("what is AI?", 123.4, "rag")
        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]
        assert item["query"] == "what is AI?"
        assert item["latency"] == 123.4
        assert item["source"] == "rag"
        mock_push.assert_called_with("QueryLatency", 123.4, unit="Milliseconds")

    def test_get_metrics_returns_list(self):
        import backend.app.metrics as metrics

        with patch.object(metrics, "_get_table") as mock_get_table, patch(
            "backend.app.metrics.push_metric"
        ):
            mock_table = MagicMock()
            mock_get_table.return_value = mock_table
            mock_table.scan.return_value = {
                "Items": [
                    {
                        "id": "1",
                        "query": "q",
                        "timestamp": "100",
                        "latency": 50,
                        "source": "rag",
                    }
                ],
                "LastEvaluatedKey": None,
            }
            result = metrics.get_metrics(window_seconds=3600)
        assert isinstance(result, list)

    def test_get_metrics_paginates(self):
        import backend.app.metrics as metrics

        with patch.object(metrics, "_get_table") as mock_get_table, patch(
            "backend.app.metrics.push_metric"
        ):
            mock_table = MagicMock()
            mock_get_table.return_value = mock_table
            mock_table.scan.side_effect = [
                {"Items": [{"id": "1"}], "LastEvaluatedKey": {"id": "1"}},
                {"Items": [{"id": "2"}], "LastEvaluatedKey": None},
            ]
            result = metrics.get_metrics()
        assert len(result) == 2

    def test_get_metrics_exception_returns_empty_and_pushes_failure(self):
        import backend.app.metrics as metrics

        with patch.object(metrics, "_get_table") as mock_get_table, patch(
            "backend.app.metrics.push_metric"
        ) as mock_push:
            mock_table = MagicMock()
            mock_get_table.return_value = mock_table
            mock_table.scan.side_effect = Exception("DynamoDB down")
            result = metrics.get_metrics()
        assert result == []
        mock_push.assert_called_with("MetricQueryFailure", 1, unit="Count")
