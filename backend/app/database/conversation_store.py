"""
SQLite-based conversation persistence for MentorAI.
Stores conversations and messages so they survive browser refresh.
"""

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationStore:
    """SQLite store for conversation history."""

    def __init__(self, db_path: str):
        """
        Initialize the conversation store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"ConversationStore initialized with database at {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id);
            """)
            conn.commit()
            logger.info("Conversation database tables initialized")
        finally:
            conn.close()

    def create_conversation(self, first_message: str) -> str:
        """
        Create a new conversation with an auto-generated title.

        Args:
            first_message: The first user message, used to generate a title

        Returns:
            The new conversation's ID
        """
        conversation_id = str(uuid.uuid4())
        title = first_message[:50].strip()
        if len(first_message) > 50:
            title += "..."
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
            conn.commit()
            logger.info(f"Created conversation {conversation_id}: {title}")
            return conversation_id
        finally:
            conn.close()

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
    ) -> str:
        """
        Add a message to a conversation.

        Args:
            conversation_id: The conversation to add the message to
            role: Message role ("user" or "assistant")
            content: Message content
            sources: Optional list of source dicts (for assistant messages)

        Returns:
            The new message's ID
        """
        message_id = str(uuid.uuid4())
        sources_json = json.dumps(sources) if sources else None
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, sources_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()
            return message_id
        finally:
            conn.close()

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """
        Get a conversation with all its messages.

        Args:
            conversation_id: The conversation ID

        Returns:
            Dict with conversation details and messages, or None if not found
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not row:
                return None

            messages = conn.execute(
                "SELECT id, role, content, sources, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()

            return {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": [
                    {
                        "id": m["id"],
                        "role": m["role"],
                        "content": m["content"],
                        "sources": json.loads(m["sources"]) if m["sources"] else None,
                        "created_at": m["created_at"],
                    }
                    for m in messages
                ],
            }
        finally:
            conn.close()

    def list_conversations(self, limit: int = 50) -> list:
        """
        List recent conversations with summary info.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation summary dicts
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count,
                    (SELECT content FROM messages
                     WHERE conversation_id = c.id
                     ORDER BY created_at ASC LIMIT 1) AS preview
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": r["message_count"],
                    "preview": (r["preview"][:100] + "...") if r["preview"] and len(r["preview"]) > 100 else r["preview"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.

        Args:
            conversation_id: The conversation ID to delete

        Returns:
            True if the conversation was found and deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted conversation {conversation_id}")
            return deleted
        finally:
            conn.close()

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Update a conversation's title.

        Args:
            conversation_id: The conversation ID
            title: The new title

        Returns:
            True if the conversation was found and updated
        """
        conn = self._get_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


# Singleton pattern (matches get_embedding_service, get_llm_service, etc.)
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """
    Get the global conversation store instance (singleton pattern).

    Returns:
        ConversationStore instance
    """
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore(settings.database_path)
    return _conversation_store


def reset_conversation_store() -> None:
    """Reset the global conversation store instance. Used for testing."""
    global _conversation_store
    _conversation_store = None
