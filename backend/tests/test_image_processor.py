"""
Tests for the image processor service.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.image_processor import (
    ImageProcessorService,
    ImageExtractionResult,
    get_image_processor,
    reset_image_processor,
    SUPPORTED_FORMATS,
    MEDIA_TYPES,
)


@pytest.mark.unit
class TestImageExtractionResult:
    """Test ImageExtractionResult class."""

    def test_valid_result(self):
        """Test a valid extraction result."""
        result = ImageExtractionResult(
            quote="The only way out is through.",
            author="Robert Frost",
            source="Waking Up",
            confidence="high",
        )
        assert result.is_valid
        assert result.quote == "The only way out is through."
        assert result.author == "Robert Frost"
        assert result.source == "Waking Up"
        assert result.error is None

    def test_invalid_result_no_quote(self):
        """Test that result with no quote is invalid."""
        result = ImageExtractionResult(
            quote=None,
            author="Some Author",
            source=None,
        )
        assert not result.is_valid

    def test_invalid_result_empty_quote(self):
        """Test that result with empty quote is invalid."""
        result = ImageExtractionResult(
            quote="   ",
            author=None,
            source=None,
        )
        assert not result.is_valid

    def test_from_error(self):
        """Test creating result from error."""
        result = ImageExtractionResult.from_error("API timeout")
        assert not result.is_valid
        assert result.error == "API timeout"
        assert result.confidence == "low"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ImageExtractionResult(
            quote="Test quote",
            author="Test Author",
            source="Test Source",
            confidence="high",
        )
        d = result.to_dict()
        assert d["quote"] == "Test quote"
        assert d["author"] == "Test Author"
        assert d["source"] == "Test Source"
        assert d["confidence"] == "high"
        assert d["error"] is None


@pytest.mark.unit
class TestImageProcessorService:
    """Test ImageProcessorService class."""

    @patch("app.services.image_processor.Anthropic")
    def test_init_with_defaults(self, mock_anthropic):
        """Test service initialization with default settings."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "claude-3-5-haiku-20241022"

            service = ImageProcessorService()
            assert service.api_key == "test-key"
            assert service.model == "claude-3-5-haiku-20241022"

    @patch("app.services.image_processor.Anthropic")
    def test_init_with_custom_params(self, mock_anthropic):
        """Test service initialization with custom parameters."""
        service = ImageProcessorService(
            api_key="custom-key",
            model="custom-model",
            rate_limit_delay=2.0,
        )
        assert service.api_key == "custom-key"
        assert service.model == "custom-model"
        assert service.rate_limit_delay == 2.0

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = None

            with pytest.raises(ValueError, match="API key not configured"):
                ImageProcessorService()

    @patch("app.services.image_processor.Anthropic")
    def test_parse_response_valid_json(self, mock_anthropic):
        """Test parsing a valid JSON response."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service = ImageProcessorService()
            response = '{"quote": "Test quote", "author": "Test Author", "source": "Test Source"}'

            result = service._parse_response(response)
            assert result.quote == "Test quote"
            assert result.author == "Test Author"
            assert result.source == "Test Source"
            assert result.confidence == "high"

    @patch("app.services.image_processor.Anthropic")
    def test_parse_response_with_surrounding_text(self, mock_anthropic):
        """Test parsing JSON with surrounding explanation text."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service = ImageProcessorService()
            response = 'Here is the extracted quote:\n{"quote": "Test", "author": null, "source": null}\nDone!'

            result = service._parse_response(response)
            assert result.quote == "Test"
            assert result.confidence == "medium"  # No author

    @patch("app.services.image_processor.Anthropic")
    def test_parse_response_no_author_medium_confidence(self, mock_anthropic):
        """Test that no author results in medium confidence."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service = ImageProcessorService()
            response = '{"quote": "Test quote", "author": null, "source": null}'

            result = service._parse_response(response)
            assert result.confidence == "medium"

    @patch("app.services.image_processor.Anthropic")
    def test_parse_response_no_quote_low_confidence(self, mock_anthropic):
        """Test that no quote results in low confidence."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service = ImageProcessorService()
            response = '{"quote": null, "author": null, "source": null}'

            result = service._parse_response(response)
            assert result.confidence == "low"

    @patch("app.services.image_processor.Anthropic")
    def test_parse_response_invalid_json(self, mock_anthropic):
        """Test parsing invalid JSON returns error result."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service = ImageProcessorService()
            response = "This is not JSON at all"

            result = service._parse_response(response)
            assert not result.is_valid
            assert result.error is not None
            assert "No JSON found" in result.error

    @patch("app.services.image_processor.Anthropic")
    def test_extract_quote_file_not_found(self, mock_anthropic):
        """Test extraction with non-existent file."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service = ImageProcessorService()
            result = service.extract_quote_from_image(Path("/nonexistent/image.jpg"))

            assert not result.is_valid
            assert "not found" in result.error

    @patch("app.services.image_processor.Anthropic")
    def test_extract_quote_unsupported_format(self, mock_anthropic, tmp_path):
        """Test extraction with unsupported file format."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            # Create a file with unsupported extension
            bad_file = tmp_path / "test.bmp"
            bad_file.write_bytes(b"fake image data")

            service = ImageProcessorService()
            result = service.extract_quote_from_image(bad_file)

            assert not result.is_valid
            assert "Unsupported image format" in result.error

    @patch("app.services.image_processor.Anthropic")
    def test_extract_quote_success(self, mock_anthropic, tmp_path):
        """Test successful quote extraction."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            # Create mock API response
            mock_response = MagicMock()
            mock_response.content = [
                MagicMock(text='{"quote": "Test quote", "author": "Test Author", "source": "Waking Up"}')
            ]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            # Create a test image file
            test_image = tmp_path / "test.jpg"
            test_image.write_bytes(b"fake jpeg data")

            service = ImageProcessorService()
            service._last_call_time = 0  # Skip rate limiting for test
            result = service.extract_quote_from_image(test_image)

            assert result.is_valid
            assert result.quote == "Test quote"
            assert result.author == "Test Author"
            assert result.source == "Waking Up"


@pytest.mark.unit
class TestSupportedFormats:
    """Test supported image format constants."""

    def test_supported_formats(self):
        """Test that common formats are supported."""
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".gif" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS

    def test_media_types(self):
        """Test media type mappings."""
        assert MEDIA_TYPES[".jpg"] == "image/jpeg"
        assert MEDIA_TYPES[".jpeg"] == "image/jpeg"
        assert MEDIA_TYPES[".png"] == "image/png"
        assert MEDIA_TYPES[".gif"] == "image/gif"
        assert MEDIA_TYPES[".webp"] == "image/webp"


@pytest.mark.unit
class TestSingleton:
    """Test singleton pattern for image processor."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_image_processor()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_image_processor()

    @patch("app.services.image_processor.Anthropic")
    def test_get_image_processor_returns_same_instance(self, mock_anthropic):
        """Test that get_image_processor returns the same instance."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service1 = get_image_processor()
            service2 = get_image_processor()
            assert service1 is service2

    @patch("app.services.image_processor.Anthropic")
    def test_reset_image_processor(self, mock_anthropic):
        """Test that reset creates a new instance."""
        with patch("app.services.image_processor.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.image_extraction_model = "test-model"

            service1 = get_image_processor()
            reset_image_processor()
            service2 = get_image_processor()
            assert service1 is not service2
