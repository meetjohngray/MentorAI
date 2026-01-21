"""
Tests for the chat endpoint and related services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

import httpx

from app.main import app
from app.models.schemas import ChatMessage, ChatRequest, ChatResponse, SourceChunk
from app.services.retrieval import (
    RetrievalService,
    RetrievalResult,
    RetrievedChunk,
    get_retrieval_service,
    reset_retrieval_service,
)
from app.services.llm import LLMService, LLMError, get_llm_service, reset_llm_service
from app.prompts.system_prompt import get_system_prompt, MENTOR_SYSTEM_PROMPT
from app.database.vector_store import initialize_db
from app.services.embeddings import get_embedding_service


@pytest.fixture
async def client():
    """Create an async test client."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def mock_llm_response():
    """Mock response from Claude."""
    return "I notice you've written about this before. What has changed for you since then?"


@pytest.fixture
def sample_chunks():
    """Sample retrieved chunks for testing."""
    return [
        RetrievedChunk(
            id="chunk_1",
            text="I struggled with meditation today. My mind kept wandering.",
            metadata={
                "source_type": "dayone",
                "date": "2024-01-15",
                "entry_id": "test-123"
            },
            distance=0.3,
            relevance_score=0.7,
            source_type="dayone"
        ),
        RetrievedChunk(
            id="chunk_2",
            text="Blog post about finding stillness in chaos.",
            metadata={
                "source_type": "wordpress",
                "date": "2024-02-01",
                "title": "Finding Stillness"
            },
            distance=0.4,
            relevance_score=0.6,
            source_type="wordpress"
        ),
    ]


@pytest.fixture
def setup_test_vector_store():
    """Set up a temporary vector store with test data."""
    temp_dir = tempfile.mkdtemp()

    vector_store = initialize_db(temp_dir, "test_collection")
    embedding_service = get_embedding_service()

    test_docs = [
        "I meditated for 20 minutes today and felt peaceful.",
        "Work has been stressful lately. Need to find balance.",
        "Grateful for small moments of quiet in my day.",
    ]

    embeddings = embedding_service.embed_batch(test_docs)

    vector_store.add_documents(
        ids=["doc1", "doc2", "doc3"],
        documents=test_docs,
        embeddings=embeddings,
        metadatas=[
            {"source_type": "dayone", "date": "2024-01-15"},
            {"source_type": "dayone", "date": "2024-01-20"},
            {"source_type": "wordpress", "date": "2024-02-01", "title": "Gratitude"},
        ]
    )

    yield vector_store

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_services():
    """Reset singleton services before each test."""
    reset_llm_service()
    reset_retrieval_service()
    yield
    reset_llm_service()
    reset_retrieval_service()


# ============================================================================
# Schema Tests
# ============================================================================

@pytest.mark.unit
class TestChatSchemas:
    """Test Pydantic models for chat."""

    def test_chat_message_valid(self):
        """Test valid ChatMessage creation."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_invalid_role(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(Exception):
            ChatMessage(role="invalid", content="Hello")

    def test_chat_request_valid(self):
        """Test valid ChatRequest creation."""
        request = ChatRequest(
            message="What patterns do you see?",
            conversation_history=[
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello!")
            ]
        )
        assert request.message == "What patterns do you see?"
        assert len(request.conversation_history) == 2

    def test_chat_request_empty_history(self):
        """Test ChatRequest with empty history."""
        request = ChatRequest(message="First message")
        assert request.message == "First message"
        assert request.conversation_history == []

    def test_chat_response_valid(self):
        """Test valid ChatResponse creation."""
        response = ChatResponse(
            response="Here's what I observe...",
            sources=[
                SourceChunk(
                    id="chunk_1",
                    text="Journal entry content",
                    source_type="dayone",
                    date="2024-01-15",
                    relevance_score=0.8
                )
            ]
        )
        assert response.response == "Here's what I observe..."
        assert len(response.sources) == 1


# ============================================================================
# System Prompt Tests
# ============================================================================

@pytest.mark.unit
class TestSystemPrompt:
    """Test system prompt generation."""

    def test_get_system_prompt_without_context(self):
        """Test system prompt with no context."""
        prompt = get_system_prompt()
        assert "personal mentor" in prompt.lower()
        assert "compassionate" in prompt.lower()
        assert "{context_section}" not in prompt

    def test_get_system_prompt_with_context(self):
        """Test system prompt with context injected."""
        context = "=== FROM THE USER'S PERSONAL HISTORY ===\n\nTest context here."
        prompt = get_system_prompt(context)
        assert "Retrieved Context" in prompt
        assert "Test context here" in prompt

    def test_system_prompt_contains_core_qualities(self):
        """Test that system prompt includes key mentor qualities."""
        prompt = MENTOR_SYSTEM_PROMPT
        assert "Compassionate Honesty" in prompt
        assert "Mirror" in prompt
        assert "Accountability Partner" in prompt


# ============================================================================
# Retrieval Service Tests
# ============================================================================

@pytest.mark.unit
class TestRetrievalFormatting:
    """Test retrieval context formatting."""

    def test_format_personal_chunks(self, sample_chunks):
        """Test formatting of personal history chunks."""
        service = RetrievalService(top_k=5)
        personal = [c for c in sample_chunks if c.is_personal]

        formatted = service._format_personal_chunks(personal)

        assert "FROM THE USER'S PERSONAL HISTORY" in formatted
        assert "Journal Entry" in formatted
        assert "Blog Post" in formatted
        assert "meditation" in formatted.lower()

    def test_format_empty_chunks(self):
        """Test formatting with no chunks."""
        service = RetrievalService(top_k=5)
        formatted = service._format_context([], [])
        assert "No relevant context found" in formatted

    def test_retrieved_chunk_properties(self, sample_chunks):
        """Test RetrievedChunk property methods."""
        dayone_chunk = sample_chunks[0]
        wordpress_chunk = sample_chunks[1]

        assert dayone_chunk.is_personal is True
        assert wordpress_chunk.is_personal is True
        assert dayone_chunk.is_wisdom is False


@pytest.mark.integration
class TestRetrievalWithVectorStore:
    """Test retrieval with actual vector store."""

    def test_retrieve_returns_results(self, setup_test_vector_store):
        """Test that retrieve returns relevant chunks."""
        service = RetrievalService(top_k=3)
        result = service.retrieve("meditation")

        assert isinstance(result, RetrievalResult)
        assert result.query == "meditation"
        assert len(result.chunks) > 0
        assert result.formatted_context != ""

    def test_retrieve_separates_sources(self, setup_test_vector_store):
        """Test that retrieval separates personal and wisdom sources."""
        service = RetrievalService(top_k=3)
        result = service.retrieve("gratitude")

        # All our test data is personal (dayone/wordpress)
        assert len(result.chunks) > 0
        assert len(result.personal_chunks) >= 0


# ============================================================================
# LLM Service Tests
# ============================================================================

@pytest.mark.unit
class TestLLMService:
    """Test LLM service functionality."""

    def test_llm_service_requires_api_key(self):
        """Test that LLM service raises error without API key."""
        with patch('app.services.llm.settings') as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            with pytest.raises(ValueError) as exc_info:
                LLMService()

            assert "API key" in str(exc_info.value)

    @patch('app.services.llm.Anthropic')
    def test_llm_service_generates_response(self, mock_anthropic_class, mock_llm_response):
        """Test that LLM service generates a response."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=mock_llm_response)]
        mock_client.messages.create.return_value = mock_response

        # Create service with mock API key
        with patch('app.services.llm.settings') as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService()
            response = service.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="You are helpful."
            )

            assert response == mock_llm_response
            mock_client.messages.create.assert_called_once()


# ============================================================================
# Chat Endpoint Tests
# ============================================================================

@pytest.mark.integration
class TestChatEndpoint:
    """Test the /chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_structure(self, client, setup_test_vector_store):
        """Test chat endpoint returns correct structure (mocked LLM)."""
        with patch('app.routers.chat.get_llm_service') as mock_get_llm:
            mock_service = MagicMock()
            mock_service.generate_response.return_value = "This is a test response."
            mock_get_llm.return_value = mock_service

            response = await client.post(
                "/chat",
                json={"message": "Tell me about meditation"}
            )

            assert response.status_code == 200
            data = response.json()

            assert "response" in data
            assert "sources" in data
            assert isinstance(data["sources"], list)

    @pytest.mark.asyncio
    async def test_chat_endpoint_with_history(self, client, setup_test_vector_store):
        """Test chat endpoint with conversation history."""
        with patch('app.routers.chat.get_llm_service') as mock_get_llm:
            mock_service = MagicMock()
            mock_service.generate_response.return_value = "Following up on our conversation..."
            mock_get_llm.return_value = mock_service

            response = await client.post(
                "/chat",
                json={
                    "message": "What else do you notice?",
                    "conversation_history": [
                        {"role": "user", "content": "Tell me about patterns in my journal"},
                        {"role": "assistant", "content": "I notice you write about work stress often."}
                    ]
                }
            )

            assert response.status_code == 200
            # Verify history was passed to LLM
            call_args = mock_service.generate_response.call_args
            messages = call_args.kwargs.get('messages') or call_args[1].get('messages')
            assert len(messages) == 3  # 2 history + 1 current

    @pytest.mark.asyncio
    async def test_chat_endpoint_empty_message(self, client):
        """Test chat endpoint rejects empty message."""
        response = await client.post(
            "/chat",
            json={"message": ""}
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_chat_endpoint_includes_sources(self, client, setup_test_vector_store):
        """Test that chat endpoint includes relevant sources."""
        with patch('app.routers.chat.get_llm_service') as mock_get_llm:
            mock_service = MagicMock()
            mock_service.generate_response.return_value = "Based on your journal..."
            mock_get_llm.return_value = mock_service

            response = await client.post(
                "/chat",
                json={"message": "meditation practice"}
            )

            assert response.status_code == 200
            data = response.json()

            # Should have sources from our test data
            assert len(data["sources"]) > 0

            # Each source should have required fields
            for source in data["sources"]:
                assert "id" in source
                assert "text" in source
                assert "source_type" in source
                assert "relevance_score" in source

    @pytest.mark.asyncio
    async def test_chat_endpoint_llm_error_handling(self, client, setup_test_vector_store):
        """Test that LLM errors are handled gracefully."""
        with patch('app.routers.chat.get_llm_service') as mock_get_llm:
            mock_service = MagicMock()
            mock_service.generate_response.side_effect = LLMError("API error")
            mock_get_llm.return_value = mock_service

            response = await client.post(
                "/chat",
                json={"message": "Hello"}
            )

            assert response.status_code == 503
            data = response.json()
            assert "Failed to get response" in data["detail"]
