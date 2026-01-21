"""
Claude API service for LLM interactions.
Handles all communication with the Anthropic API.
"""

import logging
from typing import List, Optional, AsyncIterator

import anthropic
from anthropic import Anthropic, APIError

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Claude API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM service.

        Args:
            api_key: Anthropic API key (uses settings if not provided)
            model: Model to use (uses settings if not provided)
        """
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY in your .env file."
            )

        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"LLM service initialized with model: {self.model}")

    def generate_response(
        self,
        messages: List[dict],
        system_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate a response from Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt to set Claude's behavior
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0-1)

        Returns:
            The generated response text
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
            )

            # Extract the text content from the response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return ""

        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise LLMError(f"Failed to generate response: {e}") from e

    async def generate_response_stream(
        self,
        messages: List[dict],
        system_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt to set Claude's behavior
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0-1)

        Yields:
            Text chunks as they are generated
        """
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except APIError as e:
            logger.error(f"Claude API streaming error: {e}")
            raise LLMError(f"Failed to stream response: {e}") from e


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance (singleton pattern).

    Returns:
        LLMService instance

    Raises:
        ValueError: If API key is not configured
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_llm_service() -> None:
    """Reset the global LLM service instance (useful for testing)."""
    global _llm_service
    _llm_service = None
