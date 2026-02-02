"""
Tests for configuration management.
"""

import pytest
from unittest.mock import patch

from app.config import Settings, ClaudeModel, MODEL_INFO


@pytest.mark.unit
class TestClaudeModel:
    """Test ClaudeModel enum."""

    def test_list_models_returns_all_values(self):
        """Test that list_models returns all model API strings."""
        models = ClaudeModel.list_models()
        assert isinstance(models, list)
        assert len(models) == len(ClaudeModel)
        assert "claude-opus-4-20250514" in models
        assert "claude-sonnet-4-20250514" in models

    def test_get_default_returns_opus(self):
        """Test that the default model is Opus 4."""
        default = ClaudeModel.get_default()
        assert default == ClaudeModel.OPUS_4


@pytest.mark.unit
class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test that default settings are sensible."""
        s = Settings(anthropic_api_key=None)
        assert s.chroma_path == "./data/chroma"
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.retrieval_top_k == 10
        assert s.chunk_target_tokens == 650
        assert s.chunk_max_tokens == 800
        assert s.cors_origins == "http://localhost:5173"

    def test_get_cors_origins_single(self):
        """Test parsing a single CORS origin."""
        s = Settings(cors_origins="http://localhost:5173")
        origins = s.get_cors_origins()
        assert origins == ["http://localhost:5173"]

    def test_get_cors_origins_multiple(self):
        """Test parsing multiple CORS origins."""
        s = Settings(cors_origins="http://localhost:5173,http://localhost:3000")
        origins = s.get_cors_origins()
        assert origins == ["http://localhost:5173", "http://localhost:3000"]

    def test_get_cors_origins_strips_whitespace(self):
        """Test that CORS origins are trimmed."""
        s = Settings(cors_origins="http://localhost:5173 , http://localhost:3000 ")
        origins = s.get_cors_origins()
        assert origins == ["http://localhost:5173", "http://localhost:3000"]

    def test_get_model_info_known_model(self):
        """Test get_model_info for a known model."""
        s = Settings(claude_model="claude-opus-4-20250514")
        info = s.get_model_info()
        assert info["name"] == "Claude Opus 4"
        assert "context_window" in info

    def test_get_model_info_unknown_model(self):
        """Test get_model_info for an unknown model string."""
        s = Settings(claude_model="claude-future-99-20260101")
        info = s.get_model_info()
        assert info["name"] == "claude-future-99-20260101"
        assert info["description"] == "Custom/new model"

    def test_set_model_with_enum(self):
        """Test set_model with a ClaudeModel enum value."""
        s = Settings()
        s.set_model(ClaudeModel.SONNET_4)
        assert s.claude_model == "claude-sonnet-4-20250514"

    def test_set_model_with_string(self):
        """Test set_model with a model API string."""
        s = Settings()
        s.set_model("claude-3-5-haiku-20241022")
        assert s.claude_model == "claude-3-5-haiku-20241022"


@pytest.mark.unit
class TestModelInfo:
    """Test MODEL_INFO dictionary."""

    def test_all_models_have_info(self):
        """Test that all Claude models have entries in MODEL_INFO."""
        for model in ClaudeModel:
            assert model in MODEL_INFO, f"Missing MODEL_INFO for {model}"

    def test_model_info_has_required_fields(self):
        """Test that each model info has required fields."""
        required_fields = {"name", "description", "context_window", "strengths", "cost_tier"}
        for model, info in MODEL_INFO.items():
            for field in required_fields:
                assert field in info, f"Missing '{field}' in MODEL_INFO for {model}"
