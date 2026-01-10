"""
Configuration management for MentorAI.
Loads settings from environment variables.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # API Keys (optional for testing)
    anthropic_api_key: Optional[str] = None

    # Paths
    database_path: str = "./data/mentor.db"
    chroma_path: str = "./data/chroma"

    # Model settings
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval settings
    retrieval_top_k: int = 10  # Number of chunks to retrieve

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


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