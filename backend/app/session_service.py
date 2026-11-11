"""
DynamoDB-backed Conversation Memory for multi-turn chat sessions.

Keyed by user_id (PK) and session_id (SK) with automatic TTL expiration.
Allows follow-up questions to retain short-term dialogue context without
re-sending entire conversations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

DYNAMODB_SESSIONS_TABLE = os.getenv("DYNAMODB_SESSIONS_TABLE", "rag-sessions")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))  # 24 hours
DEFAULT_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "5"))

_dynamodb = None
_table = None


def _get_table():
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb")
        _table = _dynamodb.Table(DYNAMODB_SESSIONS_TABLE)
    return _table


def get_session_history(
    user_id: str,
    session_id: str,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> List[dict]:
    """
    Retrieve recent conversation turns for a user session.

    Returns:
        List of dicts: [{"role": "user"|"assistant", "content": str, "timestamp": int}]
    """
    if not user_id or not session_id:
        return []

    try:
        response = _get_table().get_item(
            Key={"user_id": str(user_id), "session_id": str(session_id)}
        )
        item = response.get("Item")
        if not item:
            return []

        raw_messages = item.get("messages", "[]")
        if isinstance(raw_messages, str):
            messages = json.loads(raw_messages)
        else:
            messages = list(raw_messages)

        # Return the last 2 * max_turns messages (user + assistant pairs)
        return messages[-(max_turns * 2) :]
    except Exception as e:
        logger.warning(
            "Failed to retrieve session history for user=%s session=%s: %s",
            user_id,
            session_id,
            e,
        )
        return []


def add_session_turn(
    user_id: str,
    session_id: str,
    query: str,
    answer: str,
) -> None:
    """
    Append a user query and assistant response turn to the session.
    """
    if not user_id or not session_id:
        return

    now = int(time.time())
    ttl = now + SESSION_TTL_SECONDS

    try:
        current_history = get_session_history(user_id, session_id, max_turns=10)
        new_turns = [
            {"role": "user", "content": query.strip(), "timestamp": now},
            {"role": "assistant", "content": answer.strip(), "timestamp": now},
        ]
        updated_messages = current_history + new_turns

        _get_table().put_item(
            Item={
                "user_id": str(user_id),
                "session_id": str(session_id),
                "messages": json.dumps(updated_messages),
                "updated_at": now,
                "ttl": ttl,
            }
        )
        logger.debug(
            "Recorded session turn for user=%s session=%s (total %d msgs)",
            user_id,
            session_id,
            len(updated_messages),
        )
    except Exception as e:
        logger.warning(
            "Failed to record session turn for user=%s session=%s: %s",
            user_id,
            session_id,
            e,
        )


def clear_session(user_id: str, session_id: str) -> bool:
    """Delete session history from DynamoDB."""
    if not user_id or not session_id:
        return False

    try:
        _get_table().delete_item(
            Key={"user_id": str(user_id), "session_id": str(session_id)}
        )
        logger.info(
            "Cleared session history for user=%s session=%s", user_id, session_id
        )
        return True
    except Exception as e:
        logger.warning(
            "Failed to clear session for user=%s session=%s: %s",
            user_id,
            session_id,
            e,
        )
        return False


def format_conversation_context(history: List[dict]) -> str:
    """Format dialogue history into a plain text prompt context."""
    if not history:
        return ""

    lines = []
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)
