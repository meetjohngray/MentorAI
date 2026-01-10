"""
Unit tests for the DayOne ingestion script.
"""

import pytest
import json
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ingest_dayone import (
    estimate_tokens,
    chunk_text,
    parse_dayone_entry,
    process_entry
)


@pytest.mark.unit
class TestEstimateTokens:
    """Test the token estimation function."""

    def test_estimate_tokens_empty_string(self):
        """Test estimating tokens for an empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test estimating tokens for short text."""
        text = "This is a test."
        # Roughly 4 chars per token
        estimated = estimate_tokens(text)
        assert estimated == len(text) // 4

    def test_estimate_tokens_long_text(self):
        """Test estimating tokens for longer text."""
        text = "a" * 1000
        estimated = estimate_tokens(text)
        assert estimated == 250  # 1000 / 4


@pytest.mark.unit
class TestChunkText:
    """Test the text chunking function."""

    def test_chunk_short_text(self):
        """Test that short text is not chunked."""
        text = "This is a short text that fits in one chunk."
        chunks = chunk_text(text, target_tokens=100, max_tokens=150)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text_by_paragraphs(self):
        """Test chunking long text by paragraphs."""
        paragraphs = [
            "Paragraph one. " * 50,
            "Paragraph two. " * 50,
            "Paragraph three. " * 50
        ]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, target_tokens=50, max_tokens=100)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, target_tokens=10, max_tokens=20)

        # Join chunks and remove extra whitespace for comparison
        reconstructed = " ".join(chunks).replace("\n\n", " ")
        original = text.replace("\n\n", " ")

        # All content should be preserved
        assert "Paragraph one" in reconstructed
        assert "Paragraph two" in reconstructed
        assert "Paragraph three" in reconstructed

    def test_chunk_very_long_paragraph(self, long_text):
        """Test chunking when a single paragraph exceeds max tokens."""
        chunks = chunk_text(long_text, target_tokens=100, max_tokens=200)

        # Should create multiple chunks even from continuous text
        assert len(chunks) > 1

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", target_tokens=100, max_tokens=150)
        # Should return empty list or list with empty string
        assert len(chunks) <= 1


@pytest.mark.unit
class TestParseDayOneEntry:
    """Test parsing a DayOne entry."""

    def test_parse_complete_entry(self):
        """Test parsing an entry with all fields."""
        entry = {
            "uuid": "TEST-123",
            "creationDate": "2024-01-15T10:30:00Z",
            "text": "Test journal entry",
            "tags": ["meditation", "mindfulness"],
            "photos": [{"identifier": "photo1.jpg"}, {"identifier": "photo2.jpg"}]
        }

        parsed = parse_dayone_entry(entry)

        assert parsed["uuid"] == "TEST-123"
        assert parsed["creation_date"] == "2024-01-15T10:30:00Z"
        assert parsed["text"] == "Test journal entry"
        assert parsed["tags"] == ["meditation", "mindfulness"]
        assert parsed["photos"] == ["photo1.jpg", "photo2.jpg"]

    def test_parse_minimal_entry(self):
        """Test parsing an entry with minimal fields."""
        entry = {}

        parsed = parse_dayone_entry(entry)

        assert parsed["uuid"] == ""
        assert parsed["creation_date"] == ""
        assert parsed["text"] == ""
        assert parsed["tags"] == []
        assert parsed["photos"] == []

    def test_parse_entry_without_photos(self):
        """Test parsing an entry without photos."""
        entry = {
            "uuid": "TEST-456",
            "creationDate": "2024-01-20T08:15:00Z",
            "text": "Entry without photos",
            "tags": ["test"]
        }

        parsed = parse_dayone_entry(entry)

        assert parsed["photos"] == []


@pytest.mark.unit
class TestProcessEntry:
    """Test processing a DayOne entry into chunks."""

    def test_process_simple_entry(self):
        """Test processing a simple entry."""
        entry_data = {
            "uuid": "TEST-789",
            "creation_date": "2024-01-15T10:30:00Z",
            "text": "This is a test entry.",
            "tags": ["test"],
            "photos": []
        }

        chunks = process_entry(entry_data, entry_index=0)

        assert len(chunks) == 1
        assert chunks[0]["id"] == "TEST-789_chunk_0"
        assert chunks[0]["text"] == "This is a test entry."
        assert chunks[0]["metadata"]["source_type"] == "dayone"
        assert chunks[0]["metadata"]["entry_id"] == "TEST-789"
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1
        assert chunks[0]["metadata"]["tags"] == "test"

    def test_process_entry_with_photos(self):
        """Test processing an entry with photos."""
        entry_data = {
            "uuid": "TEST-890",
            "creation_date": "2024-01-20T08:15:00Z",
            "text": "Entry with photos",
            "tags": [],
            "photos": ["photo1.jpg", "photo2.jpg"]
        }

        chunks = process_entry(entry_data, entry_index=5)

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["has_photos"] is True
        assert chunks[0]["metadata"]["photo_count"] == 2
        assert chunks[0]["metadata"]["entry_index"] == 5

    def test_process_empty_entry(self):
        """Test processing an entry with empty text."""
        entry_data = {
            "uuid": "TEST-EMPTY",
            "creation_date": "2024-02-01T19:45:00Z",
            "text": "",
            "tags": [],
            "photos": []
        }

        chunks = process_entry(entry_data, entry_index=0)

        # Empty entries should return no chunks
        assert len(chunks) == 0

    def test_process_entry_with_whitespace_only(self):
        """Test processing an entry with only whitespace."""
        entry_data = {
            "uuid": "TEST-WHITESPACE",
            "creation_date": "2024-02-01T19:45:00Z",
            "text": "   \n\n  \t  ",
            "tags": [],
            "photos": []
        }

        chunks = process_entry(entry_data, entry_index=0)

        # Whitespace-only entries should return no chunks
        assert len(chunks) == 0

    def test_process_long_entry(self, long_text):
        """Test processing a long entry that needs chunking."""
        entry_data = {
            "uuid": "TEST-LONG",
            "creation_date": "2024-03-10T11:00:00Z",
            "text": long_text,
            "tags": ["long", "test"],
            "photos": []
        }

        chunks = process_entry(entry_data, entry_index=10)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should have correct metadata
        for i, chunk in enumerate(chunks):
            assert chunk["id"] == f"TEST-LONG_chunk_{i}"
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)
            assert chunk["metadata"]["entry_index"] == 10

    def test_process_entry_with_multiple_tags(self):
        """Test processing an entry with multiple tags."""
        entry_data = {
            "uuid": "TEST-TAGS",
            "creation_date": "2024-03-25T16:30:00Z",
            "text": "Entry with many tags",
            "tags": ["tag1", "tag2", "tag3", "tag4"],
            "photos": []
        }

        chunks = process_entry(entry_data, entry_index=0)

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["tags"] == "tag1,tag2,tag3,tag4"

    def test_chunk_ids_are_unique(self, long_text):
        """Test that chunk IDs are unique."""
        entry_data = {
            "uuid": "TEST-UNIQUE",
            "creation_date": "2024-01-15T10:30:00Z",
            "text": long_text,
            "tags": [],
            "photos": []
        }

        chunks = process_entry(entry_data, entry_index=0)

        chunk_ids = [chunk["id"] for chunk in chunks]
        # All IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))
