"""
Tests for the LLM service.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.llm import (
    LLMService,
    LLMError,
    get_llm_service,
    reset_llm_service,
)
from anthropic import APIError


@pytest.fixture(autouse=True)
def reset_service():
    """Reset singleton before/after each test."""
    reset_llm_service()
    yield
    reset_llm_service()


@pytest.mark.unit
class TestLLMServiceInit:
    """Test LLM service initialization."""

    def test_requires_api_key(self):
        """Test that LLM service raises error without API key."""
        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            with pytest.raises(ValueError, match="API key"):
                LLMService()

    @patch("app.services.llm.Anthropic")
    def test_uses_custom_api_key(self, mock_anthropic_class):
        """Test that a provided API key takes precedence."""
        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "settings-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService(api_key="custom-key")
            assert service.api_key == "custom-key"

    @patch("app.services.llm.Anthropic")
    def test_uses_custom_model(self, mock_anthropic_class):
        """Test that a provided model takes precedence."""
        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-opus-4-20250514"

            service = LLMService(model="claude-sonnet-4-20250514")
            assert service.model == "claude-sonnet-4-20250514"


@pytest.mark.unit
class TestGenerateResponse:
    """Test generate_response method."""

    @patch("app.services.llm.Anthropic")
    def test_generates_response(self, mock_anthropic_class):
        """Test successful response generation."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_client.messages.create.return_value = mock_response

        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService()
            result = service.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="Be helpful.",
            )

            assert result == "Test response"
            mock_client.messages.create.assert_called_once()

    @patch("app.services.llm.Anthropic")
    def test_returns_empty_for_no_content(self, mock_anthropic_class):
        """Test that empty response content returns empty string."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService()
            result = service.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="Be helpful.",
            )

            assert result == ""

    @patch("app.services.llm.Anthropic")
    def test_wraps_api_error(self, mock_anthropic_class):
        """Test that APIError is wrapped in LLMError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_client.messages.create.side_effect = APIError(
            message="Rate limit",
            request=MagicMock(),
            body=None,
        )

        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService()
            with pytest.raises(LLMError, match="Failed to generate response"):
                service.generate_response(
                    messages=[{"role": "user", "content": "Hello"}],
                    system_prompt="Be helpful.",
                )


@pytest.mark.unit
class TestGenerateResponseStream:
    """Test generate_response_stream method."""

    @patch("app.services.llm.Anthropic")
    def test_streams_text(self, mock_anthropic_class):
        """Test that streaming yields text chunks."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.text_stream = ["Hello", " ", "world"]
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream

        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService()
            chunks = list(
                service.generate_response_stream(
                    messages=[{"role": "user", "content": "Hello"}],
                    system_prompt="Be helpful.",
                )
            )

            assert chunks == ["Hello", " ", "world"]

    @patch("app.services.llm.Anthropic")
    def test_stream_wraps_api_error(self, mock_anthropic_class):
        """Test that APIError during streaming is wrapped in LLMError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_client.messages.stream.side_effect = APIError(
            message="Streaming error",
            request=MagicMock(),
            body=None,
        )

        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service = LLMService()
            with pytest.raises(LLMError, match="Failed to stream response"):
                list(
                    service.generate_response_stream(
                        messages=[{"role": "user", "content": "Hello"}],
                        system_prompt="Be helpful.",
                    )
                )


@pytest.mark.unit
class TestSingleton:
    """Test singleton pattern."""

    @patch("app.services.llm.Anthropic")
    def test_get_llm_service_returns_singleton(self, mock_anthropic_class):
        """Test that get_llm_service returns the same instance."""
        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service1 = get_llm_service()
            service2 = get_llm_service()
            assert service1 is service2

    @patch("app.services.llm.Anthropic")
    def test_reset_clears_singleton(self, mock_anthropic_class):
        """Test that reset_llm_service clears the singleton."""
        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.claude_model = "claude-sonnet-4-20250514"

            service1 = get_llm_service()
            reset_llm_service()
            service2 = get_llm_service()
            assert service1 is not service2
