import json
import sys
from unittest.mock import MagicMock, patch  # import from unittest.mock only

# Mock google.genai before any import so embeddings module loads cleanly
_mock_genai = MagicMock()
_mock_genai_client_instance = MagicMock()
_mock_genai.Client.return_value = _mock_genai_client_instance
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", _mock_genai)


class TestWorkerLambdaHandler:
    def test_processes_single_record(self):
        with patch("backend.app.worker_lambda.process_file_from_s3") as mock_proc:
            mock_proc.return_value = {"status": "processed", "chunks": 3}
            from backend.app.worker_lambda import handler

            event = {
                "Records": [
                    {
                        "body": json.dumps(
                            {"bucket": "my-bucket", "key": "user/file.txt"}
                        )
                    }
                ]
            }
            result = handler(event, {})
        assert result["processed"] == 1
        mock_proc.assert_called_once_with("my-bucket", "user/file.txt")

    def test_processes_multiple_records(self):
        with patch("backend.app.worker_lambda.process_file_from_s3") as mock_proc:
            mock_proc.return_value = {"status": "processed"}
            from backend.app.worker_lambda import handler

            event = {
                "Records": [
                    {"body": json.dumps({"bucket": "b", "key": "k1"})},
                    {"body": json.dumps({"bucket": "b", "key": "k2"})},
                ]
            }
            result = handler(event, {})
        assert result["processed"] == 2

    def test_empty_records_returns_zero(self):
        with patch("backend.app.worker_lambda.process_file_from_s3") as mock_proc:
            from backend.app.worker_lambda import handler

            result = handler({"Records": []}, {})
        assert result["processed"] == 0
        mock_proc.assert_not_called()