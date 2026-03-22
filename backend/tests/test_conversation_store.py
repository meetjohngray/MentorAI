"""
Tests for the SQLite conversation store.
"""

import pytest
import tempfile
import os
from pathlib import Path

from app.database.conversation_store import (
    ConversationStore,
    get_conversation_store,
    reset_conversation_store,
)


@pytest.fixture
def store(temp_dir):
    """Create a ConversationStore with a temp database."""
    db_path = str(temp_dir / "test.db")
    s = ConversationStore(db_path)
    s.init_db()
    return s


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton between tests."""
    reset_conversation_store()
    yield
    reset_conversation_store()


# ============================================================================
# Database Initialization
# ============================================================================

@pytest.mark.unit
class TestInitDb:
    """Test database initialization."""

    def test_init_creates_tables(self, store):
        """Test that init_db creates the conversations and messages tables."""
        conn = store._get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        conn.close()

        assert "conversations" in table_names
        assert "messages" in table_names

    def test_init_is_idempotent(self, store):
        """Test that calling init_db twice doesn't fail."""
        store.init_db()  # second call should be fine

    def test_creates_parent_directory(self, temp_dir):
        """Test that the store creates parent directories."""
        db_path = str(temp_dir / "nested" / "dir" / "test.db")
        s = ConversationStore(db_path)
        s.init_db()
        assert Path(db_path).parent.exists()


# ============================================================================
# CRUD Operations
# ============================================================================

@pytest.mark.unit
class TestCreateConversation:
    """Test conversation creation."""

    def test_create_returns_id(self, store):
        """Test that create_conversation returns a UUID string."""
        cid = store.create_conversation("Hello world")
        assert isinstance(cid, str)
        assert len(cid) == 36  # UUID format

    def test_create_generates_title_from_message(self, store):
        """Test that title is auto-generated from first message."""
        cid = store.create_conversation("What patterns do you see in my journal?")
        conv = store.get_conversation(cid)
        assert conv["title"] == "What patterns do you see in my journal?"

    def test_create_truncates_long_title(self, store):
        """Test that long messages produce truncated titles."""
        long_msg = "A" * 100
        cid = store.create_conversation(long_msg)
        conv = store.get_conversation(cid)
        assert conv["title"] == "A" * 50 + "..."

    def test_create_short_message_no_ellipsis(self, store):
        """Test that short messages don't get ellipsis."""
        cid = store.create_conversation("Short")
        conv = store.get_conversation(cid)
        assert conv["title"] == "Short"


@pytest.mark.unit
class TestAddMessage:
    """Test adding messages to conversations."""

    def test_add_user_message(self, store):
        """Test adding a user message."""
        cid = store.create_conversation("Hello")
        mid = store.add_message(cid, "user", "Hello mentor")
        assert isinstance(mid, str)

        conv = store.get_conversation(cid)
        assert len(conv["messages"]) == 1
        assert conv["messages"][0]["role"] == "user"
        assert conv["messages"][0]["content"] == "Hello mentor"

    def test_add_assistant_message_with_sources(self, store):
        """Test adding an assistant message with sources."""
        cid = store.create_conversation("Hello")
        sources = [
            {"id": "chunk_1", "text": "Journal entry", "source_type": "dayone", "relevance_score": 0.8}
        ]
        store.add_message(cid, "assistant", "I see a pattern...", sources)

        conv = store.get_conversation(cid)
        assert len(conv["messages"]) == 1
        msg = conv["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["sources"] is not None
        assert len(msg["sources"]) == 1
        assert msg["sources"][0]["id"] == "chunk_1"

    def test_add_message_updates_conversation_timestamp(self, store):
        """Test that adding a message updates the conversation's updated_at."""
        cid = store.create_conversation("Hello")
        conv_before = store.get_conversation(cid)

        store.add_message(cid, "user", "Follow up")
        conv_after = store.get_conversation(cid)

        assert conv_after["updated_at"] >= conv_before["updated_at"]

    def test_messages_ordered_by_creation(self, store):
        """Test that messages come back in creation order."""
        cid = store.create_conversation("Hello")
        store.add_message(cid, "user", "First")
        store.add_message(cid, "assistant", "Second")
        store.add_message(cid, "user", "Third")

        conv = store.get_conversation(cid)
        contents = [m["content"] for m in conv["messages"]]
        assert contents == ["First", "Second", "Third"]


@pytest.mark.unit
class TestGetConversation:
    """Test getting conversations."""

    def test_get_nonexistent_returns_none(self, store):
        """Test that getting a nonexistent conversation returns None."""
        assert store.get_conversation("nonexistent-id") is None

    def test_get_returns_all_fields(self, store):
        """Test that get_conversation returns all expected fields."""
        cid = store.create_conversation("Test message")
        conv = store.get_conversation(cid)

        assert "id" in conv
        assert "title" in conv
        assert "created_at" in conv
        assert "updated_at" in conv
        assert "messages" in conv


@pytest.mark.unit
class TestListConversations:
    """Test listing conversations."""

    def test_list_empty(self, store):
        """Test listing when there are no conversations."""
        result = store.list_conversations()
        assert result == []

    def test_list_returns_conversations(self, store):
        """Test listing returns created conversations."""
        store.create_conversation("First")
        store.create_conversation("Second")

        result = store.list_conversations()
        assert len(result) == 2

    def test_list_ordered_by_updated_at_desc(self, store):
        """Test that list is ordered by most recently updated first."""
        cid1 = store.create_conversation("First")
        store.create_conversation("Second")

        # Update the first conversation to make it most recent
        store.add_message(cid1, "user", "Update")

        result = store.list_conversations()
        assert result[0]["title"] == "First"

    def test_list_includes_message_count(self, store):
        """Test that list includes message count."""
        cid = store.create_conversation("Hello")
        store.add_message(cid, "user", "Message 1")
        store.add_message(cid, "assistant", "Response 1")

        result = store.list_conversations()
        assert result[0]["message_count"] == 2

    def test_list_includes_preview(self, store):
        """Test that list includes a preview of the first message."""
        cid = store.create_conversation("Hello")
        store.add_message(cid, "user", "My first message about meditation")

        result = store.list_conversations()
        assert "meditation" in result[0]["preview"]

    def test_list_truncates_long_preview(self, store):
        """Test that preview is truncated to 100 chars."""
        cid = store.create_conversation("Hello")
        store.add_message(cid, "user", "X" * 200)

        result = store.list_conversations()
        assert len(result[0]["preview"]) == 103  # 100 + "..."

    def test_list_respects_limit(self, store):
        """Test that list respects the limit parameter."""
        for i in range(5):
            store.create_conversation(f"Conversation {i}")

        result = store.list_conversations(limit=3)
        assert len(result) == 3


@pytest.mark.unit
class TestDeleteConversation:
    """Test deleting conversations."""

    def test_delete_existing(self, store):
        """Test deleting an existing conversation."""
        cid = store.create_conversation("To delete")
        store.add_message(cid, "user", "A message")

        assert store.delete_conversation(cid) is True
        assert store.get_conversation(cid) is None

    def test_delete_nonexistent_returns_false(self, store):
        """Test that deleting a nonexistent conversation returns False."""
        assert store.delete_conversation("nonexistent") is False

    def test_delete_cascades_to_messages(self, store):
        """Test that deleting a conversation also deletes its messages."""
        cid = store.create_conversation("To delete")
        store.add_message(cid, "user", "Message")

        store.delete_conversation(cid)

        # Verify messages are gone too
        conn = store._get_connection()
        count = conn.execute(
            "SELECT COUNT(*) as c FROM messages WHERE conversation_id = ?", (cid,)
        ).fetchone()["c"]
        conn.close()
        assert count == 0


@pytest.mark.unit
class TestUpdateTitle:
    """Test updating conversation titles."""

    def test_update_title(self, store):
        """Test updating a conversation's title."""
        cid = store.create_conversation("Original title")
        assert store.update_conversation_title(cid, "New title") is True

        conv = store.get_conversation(cid)
        assert conv["title"] == "New title"

    def test_update_nonexistent_returns_false(self, store):
        """Test that updating a nonexistent conversation returns False."""
        assert store.update_conversation_title("nonexistent", "Title") is False


# ============================================================================
# Singleton Tests
# ============================================================================

@pytest.mark.unit
class TestSingleton:
    """Test singleton pattern."""

    def test_get_returns_instance(self, monkeypatch, temp_dir):
        """Test that get_conversation_store returns an instance."""
        monkeypatch.setattr("app.database.conversation_store.settings.database_path", str(temp_dir / "singleton.db"))
        store = get_conversation_store()
        assert isinstance(store, ConversationStore)

    def test_get_returns_same_instance(self, monkeypatch, temp_dir):
        """Test that get_conversation_store returns the same instance."""
        monkeypatch.setattr("app.database.conversation_store.settings.database_path", str(temp_dir / "singleton.db"))
        store1 = get_conversation_store()
        store2 = get_conversation_store()
        assert store1 is store2

    def test_reset_clears_instance(self, monkeypatch, temp_dir):
        """Test that reset_conversation_store clears the singleton."""
        monkeypatch.setattr("app.database.conversation_store.settings.database_path", str(temp_dir / "singleton.db"))
        store1 = get_conversation_store()
        reset_conversation_store()
        store2 = get_conversation_store()
        assert store1 is not store2
