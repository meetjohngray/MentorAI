"""
Chat router for MentorAI.
Handles the main chat endpoint with RAG integration.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.database.conversation_store import get_conversation_store
from app.models.schemas import ChatMessage, ChatRequest, ChatResponse, SourceChunk
from app.services.llm import get_llm_service, LLMError
from app.services.retrieval import get_retrieval_service, RetrievedChunk
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
    5. Persists messages to the conversation store

    Args:
        request: Chat request with message, optional conversation_id, and history

    Returns:
        ChatResponse with the mentor's response, sources, and conversation_id
    """
    logger.info(f"Chat request: {request.message[:50]}...")

    try:
        store = get_conversation_store()

        # Step 1: Resolve conversation — create if needed
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = store.create_conversation(request.message)

        # Step 2: Load history from DB if conversation_id provided but no client history
        conversation_history = request.conversation_history
        if request.conversation_id and not conversation_history:
            conversation = store.get_conversation(conversation_id)
            if conversation:
                conversation_history = [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in conversation["messages"]
                ]

        # Step 3: Save user message
        store.add_message(conversation_id, "user", request.message)

        # Step 4: Retrieve relevant context
        retrieval_service = get_retrieval_service()
        retrieval_result = retrieval_service.retrieve(request.message)

        logger.info(
            f"Retrieved {len(retrieval_result.chunks)} chunks "
            f"({len(retrieval_result.personal_chunks)} personal, "
            f"{len(retrieval_result.wisdom_chunks)} wisdom, "
            f"{len(retrieval_result.commonplace_chunks)} commonplace)"
        )

        # Step 5: Build the system prompt with context
        system_prompt = get_system_prompt(retrieval_result.formatted_context)

        # Step 6: Build the message list for Claude
        messages = _build_messages(conversation_history, request.message)

        # Step 7: Get response from Claude
        llm_service = get_llm_service()
        response_text = llm_service.generate_response(
            messages=messages,
            system_prompt=system_prompt
        )

        # Step 8: Format the sources
        sources = _format_sources(retrieval_result.chunks)

        # Step 9: Save assistant response with sources
        sources_dicts = [s.model_dump() for s in sources] if sources else None
        store.add_message(conversation_id, "assistant", response_text, sources_dicts)

        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=conversation_id,
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
    conversation_history: List[ChatMessage],
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


def _format_sources(chunks: List[RetrievedChunk]) -> List[SourceChunk]:
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
            relevance_score=chunk.relevance_score,
            tradition=chunk.metadata.get("tradition"),
            teacher=chunk.metadata.get("teacher"),
            text_title=chunk.metadata.get("text_title"),
            author=chunk.metadata.get("author"),
            book_title=chunk.metadata.get("book_title"),
        )
        sources.append(source)

    return sources
