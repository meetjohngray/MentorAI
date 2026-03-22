"""
Tests for the conversations API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

import httpx

from app.main import app
from app.database.conversation_store import (
    ConversationStore,
    get_conversation_store,
    reset_conversation_store,
)


@pytest.fixture
async def client():
    """Create an async test client."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def conversation_store(temp_dir):
    """Create and initialize a temp conversation store."""
    db_path = str(temp_dir / "test_api.db")
    store = ConversationStore(db_path)
    store.init_db()
    return store


@pytest.fixture(autouse=True)
def mock_store(conversation_store):
    """Replace the singleton store with our test store."""
    with patch("app.routers.conversations.get_conversation_store", return_value=conversation_store):
        yield conversation_store


# ============================================================================
# GET /conversations
# ============================================================================

@pytest.mark.integration
class TestListConversations:
    """Test GET /conversations endpoint."""

    @pytest.mark.asyncio
    async def test_list_empty(self, client, mock_store):
        """Test listing when no conversations exist."""
        response = await client.get("/conversations")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_with_conversations(self, client, mock_store):
        """Test listing existing conversations."""
        mock_store.create_conversation("First conversation")
        mock_store.create_conversation("Second conversation")

        response = await client.get("/conversations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_list_with_limit(self, client, mock_store):
        """Test listing with a limit parameter."""
        for i in range(5):
            mock_store.create_conversation(f"Conversation {i}")

        response = await client.get("/conversations", params={"limit": 2})
        assert response.status_code == 200
        assert len(response.json()) == 2

    @pytest.mark.asyncio
    async def test_list_returns_required_fields(self, client, mock_store):
        """Test that list returns all required fields."""
        cid = mock_store.create_conversation("Test")
        mock_store.add_message(cid, "user", "Hello")

        response = await client.get("/conversations")
        data = response.json()[0]
        assert "id" in data
        assert "title" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "message_count" in data


# ============================================================================
# GET /conversations/{id}
# ============================================================================

@pytest.mark.integration
class TestGetConversation:
    """Test GET /conversations/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_existing(self, client, mock_store):
        """Test getting an existing conversation."""
        cid = mock_store.create_conversation("Test conversation")
        mock_store.add_message(cid, "user", "Hello")
        mock_store.add_message(cid, "assistant", "Hi there")

        response = await client.get(f"/conversations/{cid}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == cid
        assert len(data["messages"]) == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, client, mock_store):
        """Test getting a nonexistent conversation."""
        response = await client.get("/conversations/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_includes_sources(self, client, mock_store):
        """Test that messages with sources include them."""
        cid = mock_store.create_conversation("Test")
        sources = [{"id": "c1", "text": "Entry", "source_type": "dayone", "relevance_score": 0.8}]
        mock_store.add_message(cid, "assistant", "Response", sources)

        response = await client.get(f"/conversations/{cid}")
        data = response.json()
        msg = data["messages"][0]
        assert msg["sources"] is not None
        assert msg["sources"][0]["id"] == "c1"


# ============================================================================
# DELETE /conversations/{id}
# ============================================================================

@pytest.mark.integration
class TestDeleteConversation:
    """Test DELETE /conversations/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, client, mock_store):
        """Test deleting an existing conversation."""
        cid = mock_store.create_conversation("To delete")
        response = await client.delete(f"/conversations/{cid}")
        assert response.status_code == 204

        # Verify it's gone
        assert mock_store.get_conversation(cid) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, client, mock_store):
        """Test deleting a nonexistent conversation."""
        response = await client.delete("/conversations/nonexistent")
        assert response.status_code == 404


# ============================================================================
# PATCH /conversations/{id}
# ============================================================================

@pytest.mark.integration
class TestUpdateConversationTitle:
    """Test PATCH /conversations/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_title(self, client, mock_store):
        """Test updating a conversation's title."""
        cid = mock_store.create_conversation("Old title")
        response = await client.patch(
            f"/conversations/{cid}",
            json={"title": "New title"},
        )
        assert response.status_code == 204

        conv = mock_store.get_conversation(cid)
        assert conv["title"] == "New title"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, client, mock_store):
        """Test updating a nonexistent conversation."""
        response = await client.patch(
            "/conversations/nonexistent",
            json={"title": "New"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_empty_title_rejected(self, client, mock_store):
        """Test that empty title is rejected."""
        cid = mock_store.create_conversation("Test")
        response = await client.patch(
            f"/conversations/{cid}",
            json={"title": ""},
        )
        assert response.status_code == 422
