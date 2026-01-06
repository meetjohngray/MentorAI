"""
Configuration management for MentorAI.
Loads settings from environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # API Keys
    anthropic_api_key: str
    
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


# Create a global settings instance
settings = Settings()