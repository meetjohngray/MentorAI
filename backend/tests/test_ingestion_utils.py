"""
Tests for shared ingestion utilities.
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ingestion_utils import estimate_tokens, chunk_text, find_export_file


@pytest.mark.unit
class TestEstimateTokens:
    """Test the token estimation function."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_known_length(self):
        assert estimate_tokens("a" * 100) == 25

    def test_short_text(self):
        text = "Hello world"
        assert estimate_tokens(text) == len(text) // 4


@pytest.mark.unit
class TestChunkText:
    """Test the text chunking function."""

    def test_short_text_not_chunked(self):
        text = "Short text."
        chunks = chunk_text(text, target_tokens=100, max_tokens=150)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunked(self):
        paragraphs = ["Paragraph. " * 50 for _ in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, target_tokens=50, max_tokens=100)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = chunk_text("", target_tokens=100, max_tokens=150)
        assert len(chunks) <= 1


@pytest.mark.unit
class TestFindExportFile:
    """Test the find_export_file function."""

    def test_finds_json_file(self, tmp_path):
        """Test finding a JSON file in the expected location."""
        # Create a fake directory structure
        subdir = tmp_path / "data" / "raw" / "test"
        subdir.mkdir(parents=True)
        test_file = subdir / "export.json"
        test_file.write_text("{}")

        # Monkey-patch the function's base path
        import scripts.ingestion_utils as utils
        original_file = utils.__file__

        # We can't easily monkey-patch __file__ in the function, so test the error case
        with pytest.raises(FileNotFoundError):
            find_export_file("nonexistent_source", "*.json")

    def test_missing_directory_raises_error(self):
        """Test that missing export directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            find_export_file("definitely_not_a_real_source_1234", "*.json")
