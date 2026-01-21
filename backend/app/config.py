"""
Configuration management for MentorAI.
Loads settings from environment variables.
"""

import os
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ClaudeModel(str, Enum):
    """Available Claude models.
    
    Update this enum as new models are released.
    Format: FRIENDLY_NAME = "api-model-string"
    """
    # Claude 4 family
    OPUS_4 = "claude-opus-4-20250514"
    SONNET_4 = "claude-sonnet-4-20250514"
    
    # Claude 3.5 family (legacy, but still available)
    SONNET_3_5 = "claude-3-5-sonnet-20241022"
    HAIKU_3_5 = "claude-3-5-haiku-20241022"
    
    @classmethod
    def list_models(cls) -> list[str]:
        """Return list of all available model API strings."""
        return [model.value for model in cls]
    
    @classmethod
    def get_default(cls) -> "ClaudeModel":
        """Return the default model."""
        return cls.OPUS_4


# Model characteristics for reference/future use
MODEL_INFO = {
    ClaudeModel.OPUS_4: {
        "name": "Claude Opus 4",
        "description": "Most capable model. Best for nuanced, emotionally intelligent conversations.",
        "context_window": 200000,
        "strengths": ["emotional nuance", "complex reasoning", "wisdom integration"],
        "cost_tier": "high",
    },
    ClaudeModel.SONNET_4: {
        "name": "Claude Sonnet 4",
        "description": "Balanced model. Good capability with faster responses.",
        "context_window": 200000,
        "strengths": ["speed", "general capability", "cost efficiency"],
        "cost_tier": "medium",
    },
    ClaudeModel.SONNET_3_5: {
        "name": "Claude Sonnet 3.5",
        "description": "Previous generation Sonnet. Still capable.",
        "context_window": 200000,
        "strengths": ["proven reliability", "good balance"],
        "cost_tier": "medium",
    },
    ClaudeModel.HAIKU_3_5: {
        "name": "Claude Haiku 3.5",
        "description": "Fast and lightweight. Good for simple queries.",
        "context_window": 200000,
        "strengths": ["speed", "low cost"],
        "cost_tier": "low",
    },
}


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # API Keys (optional for testing)
    anthropic_api_key: Optional[str] = None

    # Paths
    database_path: str = "./data/mentor.db"
    chroma_path: str = "./data/chroma"

    # Model settings
    # Can be set via CLAUDE_MODEL env var, e.g.: CLAUDE_MODEL=claude-sonnet-4-20250514
    claude_model: str = Field(
        default=ClaudeModel.get_default().value,
        description="Claude model to use for chat responses"
    )
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval settings
    retrieval_top_k: int = 10  # Number of chunks to retrieve
    
    def get_model_info(self) -> dict:
        """Get information about the currently configured model."""
        try:
            model_enum = ClaudeModel(self.claude_model)
            return MODEL_INFO.get(model_enum, {"name": self.claude_model, "description": "Unknown model"})
        except ValueError:
            # Model string not in enum (possibly a new model)
            return {"name": self.claude_model, "description": "Custom/new model"}
    
    def set_model(self, model: ClaudeModel | str) -> None:
        """Change the active model.
        
        Args:
            model: Either a ClaudeModel enum or a model API string
        """
        if isinstance(model, ClaudeModel):
            self.claude_model = model.value
        else:
            self.claude_model = model


def _get_env_file_path() -> Optional[str]:
    """Find the .env file, checking both current dir and backend dir."""
    # Check current directory
    if os.path.exists(".env"):
        return ".env"
    # Check backend directory (when running from project root)
    if os.path.exists("backend/.env"):
        return "backend/.env"
    return None


# Create a global settings instance
settings = Settings(_env_file=_get_env_file_path())
