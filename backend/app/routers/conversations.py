"""
Conversations router for MentorAI.
Handles CRUD operations for conversation persistence.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.database.conversation_store import get_conversation_store
from app.models.schemas import (
    ConversationDetail,
    ConversationSummary,
    ConversationTitleUpdate,
    ChatMessageWithSources,
    SourceChunk,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = Query(50, ge=1, le=200, description="Maximum conversations to return"),
) -> List[ConversationSummary]:
    """List recent conversations with summary info."""
    store = get_conversation_store()
    conversations = store.list_conversations(limit=limit)
    return [ConversationSummary(**c) for c in conversations]


@router.post("", response_model=ConversationSummary, status_code=201)
async def create_conversation(
    first_message: str = Query(..., min_length=1, description="First message to generate title"),
) -> ConversationSummary:
    """Create a new empty conversation."""
    store = get_conversation_store()
    conversation_id = store.create_conversation(first_message)
    conversation = store.get_conversation(conversation_id)
    return ConversationSummary(
        id=conversation["id"],
        title=conversation["title"],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
        message_count=0,
        preview=None,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    """Get a conversation with all its messages."""
    store = get_conversation_store()
    conversation = store.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    for m in conversation["messages"]:
        sources = None
        if m["sources"]:
            sources = [SourceChunk(**s) for s in m["sources"]]
        messages.append(
            ChatMessageWithSources(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                sources=sources,
                created_at=m["created_at"],
            )
        )

    return ConversationDetail(
        id=conversation["id"],
        title=conversation["title"],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
        messages=messages,
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation and all its messages."""
    store = get_conversation_store()
    deleted = store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.patch("/{conversation_id}", status_code=204)
async def update_conversation_title(
    conversation_id: str,
    body: ConversationTitleUpdate,
) -> None:
    """Update a conversation's title."""
    store = get_conversation_store()
    updated = store.update_conversation_title(conversation_id, body.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
