"""
Pydantic models for MentorAI API request/response schemas.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str = Field(..., min_length=1, description="The user's message")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
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


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str = Field(..., description="The mentor's response")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="Retrieved chunks used for context"
    )


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
