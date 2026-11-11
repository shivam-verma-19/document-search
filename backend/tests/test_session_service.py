"""
Tests for session_service.py — DynamoDB conversation memory.
"""

import sys
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws


@pytest.fixture(autouse=True)
def _evict_stubs():
    for mod in ["backend.app.session_service"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.session_service"]:
        sys.modules.pop(mod, None)


def _setup_session_table():
    db = boto3.resource("dynamodb", region_name="us-east-1")
    db.create_table(
        TableName="rag-sessions",
        KeySchema=[
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "session_id", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "session_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


class TestSessionService:
    @mock_aws
    def test_empty_session_returns_empty_list(self):
        _setup_session_table()
        import importlib

        import backend.app.session_service as ss

        importlib.reload(ss)

        history = ss.get_session_history("user1", "session1")
        assert history == []

    @mock_aws
    def test_add_and_retrieve_turn(self):
        _setup_session_table()
        import importlib

        import backend.app.session_service as ss

        importlib.reload(ss)

        ss.add_session_turn("user1", "session1", "Hello RAG", "Hello! How can I help?")
        history = ss.get_session_history("user1", "session1")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello RAG"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello! How can I help?"

    @mock_aws
    def test_multiple_turns_ordered(self):
        _setup_session_table()
        import importlib

        import backend.app.session_service as ss

        importlib.reload(ss)

        ss.add_session_turn("user1", "session1", "Q1", "A1")
        ss.add_session_turn("user1", "session1", "Q2", "A2")

        history = ss.get_session_history("user1", "session1", max_turns=5)
        assert len(history) == 4
        assert [m["content"] for m in history] == ["Q1", "A1", "Q2", "A2"]

    @mock_aws
    def test_clear_session(self):
        _setup_session_table()
        import importlib

        import backend.app.session_service as ss

        importlib.reload(ss)

        ss.add_session_turn("user1", "session1", "Q1", "A1")
        assert len(ss.get_session_history("user1", "session1")) == 2

        assert ss.clear_session("user1", "session1") is True
        assert len(ss.get_session_history("user1", "session1")) == 0

    def test_format_conversation_context(self):
        import backend.app.session_service as ss

        history = [
            {"role": "user", "content": "What is hybrid search?"},
            {"role": "assistant", "content": "It combines vectors and BM25."},
        ]
        context = ss.format_conversation_context(history)
        assert "User: What is hybrid search?" in context
        assert "Assistant: It combines vectors and BM25." in context

    def test_format_empty_returns_empty_string(self):
        import backend.app.session_service as ss

        assert ss.format_conversation_context([]) == ""
