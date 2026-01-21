"""
Chat router for MentorAI.
Handles the main chat endpoint with RAG integration.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.llm import get_llm_service, LLMError
from app.services.retrieval import get_retrieval_service
from app.prompts.system_prompt import get_system_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the MentorAI companion.

    This endpoint:
    1. Retrieves relevant context from the user's personal history
    2. Constructs a prompt with the system prompt and context
    3. Sends the conversation to Claude
    4. Returns the response with sources

    Args:
        request: Chat request with message and conversation history

    Returns:
        ChatResponse with the mentor's response and sources used
    """
    logger.info(f"Chat request: {request.message[:50]}...")

    try:
        # Step 1: Retrieve relevant context
        retrieval_service = get_retrieval_service()
        retrieval_result = retrieval_service.retrieve(request.message)

        logger.info(
            f"Retrieved {len(retrieval_result.chunks)} chunks "
            f"({len(retrieval_result.personal_chunks)} personal, "
            f"{len(retrieval_result.wisdom_chunks)} wisdom)"
        )

        # Step 2: Build the system prompt with context
        system_prompt = get_system_prompt(retrieval_result.formatted_context)

        # Step 3: Build the message list for Claude
        messages = _build_messages(request.conversation_history, request.message)

        # Step 4: Get response from Claude
        llm_service = get_llm_service()
        response_text = llm_service.generate_response(
            messages=messages,
            system_prompt=system_prompt
        )

        # Step 5: Format the sources
        sources = _format_sources(retrieval_result.chunks)

        return ChatResponse(
            response=response_text,
            sources=sources
        )

    except LLMError as e:
        logger.error(f"LLM error in chat: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to get response from AI service: {str(e)}"
        )
    except ValueError as e:
        # Likely API key not configured
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )


def _build_messages(
    conversation_history: List,
    current_message: str
) -> List[dict]:
    """
    Build the messages list for Claude API.

    Args:
        conversation_history: Previous messages in the conversation
        current_message: The current user message

    Returns:
        List of message dicts for Claude API
    """
    messages = []

    # Add conversation history
    for msg in conversation_history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": current_message
    })

    return messages


def _format_sources(chunks: List) -> List[SourceChunk]:
    """
    Format retrieved chunks as source objects.

    Args:
        chunks: List of RetrievedChunk objects

    Returns:
        List of SourceChunk objects for the response
    """
    sources = []

    for chunk in chunks:
        source = SourceChunk(
            id=chunk.id,
            text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
            source_type=chunk.source_type,
            date=chunk.metadata.get("date"),
            title=chunk.metadata.get("title"),
            relevance_score=chunk.relevance_score
        )
        sources.append(source)

    return sources
