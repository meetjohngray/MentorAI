"""
Pydantic models for MentorAI API request/response schemas.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str = Field(..., min_length=1, description="The user's message")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID to continue. None creates a new conversation."
    )
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        max_length=100,
        description="Previous messages in the conversation"
    )


class SourceChunk(BaseModel):
    """A chunk of source material used for context."""
    id: str
    text: str
    source_type: str
    date: Optional[str] = None
    title: Optional[str] = None
    relevance_score: float
    # Wisdom-specific fields
    tradition: Optional[str] = None
    teacher: Optional[str] = None
    text_title: Optional[str] = None
    # Commonplace-specific fields
    author: Optional[str] = None
    book_title: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str = Field(..., description="The mentor's response")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="Retrieved chunks used for context"
    )
    conversation_id: str = Field(..., description="The conversation ID")


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str
    version: str
    components: Dict[str, str]
    vector_store_documents: int


class SearchResult(BaseModel):
    """A single search result."""
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    relevance_score: float


class SearchResponse(BaseModel):
    """Response from the search endpoint."""
    query: str
    num_results: int
    results: List[SearchResult]


# ============================================================================
# Conversation Types
# ============================================================================


class ChatMessageWithSources(BaseModel):
    """A message that may include sources (for assistant messages)."""
    id: str
    role: Literal["user", "assistant"]
    content: str
    sources: Optional[List[SourceChunk]] = None
    created_at: datetime


class ConversationSummary(BaseModel):
    """Summary of a conversation for list view."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    preview: Optional[str] = None


class ConversationDetail(BaseModel):
    """Full conversation with messages."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageWithSources]


class ConversationTitleUpdate(BaseModel):
    """Request body for updating a conversation title."""
    title: str = Field(..., min_length=1, max_length=200)
