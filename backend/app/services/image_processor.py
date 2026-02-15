"""
Image processing service for extracting text from quote images.

Uses Claude's vision capability to extract quote text and metadata
from images in the Commonplace Book photos folder.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

from anthropic import Anthropic, APIError

from app.config import settings

logger = logging.getLogger(__name__)


# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Media types for Claude Vision API
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# System prompt for quote extraction
EXTRACTION_PROMPT = """You are an expert at extracting text from images of quotes.

Given an image, extract:
1. The exact quote text (preserve line breaks and formatting where meaningful)
2. The author's name (if shown)
3. The source/app name (if shown, e.g., "Waking Up", "Daily Stoic", "Sam Harris")

Return ONLY valid JSON in this exact format:
{
  "quote": "The extracted quote text...",
  "author": "Author Name",
  "source": "Source Name"
}

Rules:
- Only extract what is clearly visible in the image
- If author or source is not visible, use null (not "null" or "Unknown")
- Do not guess or add information not present in the image
- If the image does not contain a quote, return: {"quote": null, "author": null, "source": null}
- Preserve the original punctuation and capitalization of the quote"""


class ImageExtractionResult:
    """Result of extracting a quote from an image."""

    def __init__(
        self,
        quote: Optional[str],
        author: Optional[str],
        source: Optional[str],
        confidence: str = "high",
        error: Optional[str] = None,
    ):
        """
        Initialize extraction result.

        Args:
            quote: The extracted quote text
            author: Author name if detected
            source: Source/app name if detected
            confidence: Extraction confidence level (high/medium/low)
            error: Error message if extraction failed
        """
        self.quote = quote
        self.author = author
        self.source = source
        self.confidence = confidence
        self.error = error

    @property
    def is_valid(self) -> bool:
        """Check if extraction produced a valid quote."""
        return self.quote is not None and len(self.quote.strip()) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "quote": self.quote,
            "author": self.author,
            "source": self.source,
            "confidence": self.confidence,
            "error": self.error,
        }

    @classmethod
    def from_error(cls, error_message: str) -> "ImageExtractionResult":
        """Create a result representing an extraction error."""
        return cls(
            quote=None,
            author=None,
            source=None,
            confidence="low",
            error=error_message,
        )


class ImageProcessorService:
    """Service for extracting quotes from images using Claude Vision."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize the image processor service.

        Args:
            api_key: Anthropic API key (uses settings if not provided)
            model: Model to use for vision (uses settings if not provided)
            rate_limit_delay: Delay in seconds between API calls
        """
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.image_extraction_model

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY in your .env file."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time: float = 0

        logger.info(f"Image processor initialized with model: {self.model}")

    def _encode_image(self, image_path: Path) -> tuple[str, str]:
        """
        Read and base64-encode an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (base64_data, media_type)

        Raises:
            ValueError: If image format is not supported
        """
        suffix = image_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

        media_type = MEDIA_TYPES[suffix]

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        return image_data, media_type

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting delay between API calls."""
        if self._last_call_time > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()

    def _parse_response(self, response_text: str) -> ImageExtractionResult:
        """
        Parse Claude's JSON response into an ImageExtractionResult.

        Args:
            response_text: Raw text response from Claude

        Returns:
            Parsed extraction result
        """
        # Try to extract JSON from the response
        # Sometimes Claude includes explanation text around the JSON
        try:
            # First try direct parsing
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(response_text[start:end])
                except json.JSONDecodeError:
                    return ImageExtractionResult.from_error(
                        f"Failed to parse JSON response: {response_text[:200]}"
                    )
            else:
                return ImageExtractionResult.from_error(
                    f"No JSON found in response: {response_text[:200]}"
                )

        quote = data.get("quote")
        author = data.get("author")
        source = data.get("source")

        # Determine confidence based on what was extracted
        if quote and author:
            confidence = "high"
        elif quote:
            confidence = "medium"
        else:
            confidence = "low"

        return ImageExtractionResult(
            quote=quote,
            author=author,
            source=source,
            confidence=confidence,
        )

    def extract_quote_from_image(self, image_path: Path) -> ImageExtractionResult:
        """
        Use Claude Vision to extract quote text and metadata from an image.

        Args:
            image_path: Path to the image file

        Returns:
            ImageExtractionResult with extracted quote, author, source, and confidence
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return ImageExtractionResult.from_error(f"Image file not found: {image_path}")

        try:
            # Encode the image
            image_data, media_type = self._encode_image(image_path)

            # Apply rate limiting
            self._apply_rate_limit()

            # Call Claude Vision API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract the quote, author, and source from this image. Return only JSON.",
                            },
                        ],
                    }
                ],
                system=EXTRACTION_PROMPT,
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
                return self._parse_response(response_text)

            return ImageExtractionResult.from_error("Empty response from Claude")

        except APIError as e:
            logger.error(f"Claude API error processing {image_path}: {e}")
            return ImageExtractionResult.from_error(f"API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return ImageExtractionResult.from_error(f"Processing error: {str(e)}")


class ImageProcessorError(Exception):
    """Custom exception for image processing errors."""

    pass


# Global instance
_image_processor: Optional[ImageProcessorService] = None


def get_image_processor() -> ImageProcessorService:
    """
    Get the global image processor instance (singleton pattern).

    Returns:
        ImageProcessorService instance

    Raises:
        ValueError: If API key is not configured
    """
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessorService()
    return _image_processor


def reset_image_processor() -> None:
    """Reset the global image processor instance (useful for testing)."""
    global _image_processor
    _image_processor = None
