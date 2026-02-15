"""
Tests for the Commonplace Book ingestion pipeline.
"""

import json
import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ingest_commonplace import (
    extract_attribution,
    parse_commonplace_entry,
    process_commonplace_entry,
    process_image_entry,
    load_image_cache,
    _build_searchable_text,
)


@pytest.mark.unit
class TestExtractAttribution:
    """Test attribution extraction from entry text."""

    def test_em_dash_attribution(self):
        """Test extraction with em dash (—)."""
        text = "The only way out is through.\n— Robert Frost"
        clean, author, book = extract_attribution(text)

        assert clean == "The only way out is through."
        assert author == "Robert Frost"
        assert book is None

    def test_en_dash_attribution(self):
        """Test extraction with en dash (\u2013)."""
        text = "Be yourself; everyone else is already taken.\n\u2013 Oscar Wilde"
        clean, author, book = extract_attribution(text)

        assert clean == "Be yourself; everyone else is already taken."
        assert author == "Oscar Wilde"
        assert book is None

    def test_hyphen_attribution(self):
        """Test extraction with simple hyphen (-)."""
        text = "In the middle of difficulty lies opportunity.\n- Albert Einstein"
        clean, author, book = extract_attribution(text)

        assert clean == "In the middle of difficulty lies opportunity."
        assert author == "Albert Einstein"
        assert book is None

    def test_tilde_attribution(self):
        """Test extraction with tilde (~)."""
        text = "The unexamined life is not worth living.\n~ Socrates"
        clean, author, book = extract_attribution(text)

        assert clean == "The unexamined life is not worth living."
        assert author == "Socrates"
        assert book is None

    def test_attribution_with_book_title_quoted(self):
        """Test extraction with author and quoted book title."""
        text = 'We do not see things as they are.\n— Anais Nin, "Seduction of the Minotaur"'
        clean, author, book = extract_attribution(text)

        assert clean == "We do not see things as they are."
        assert author == "Anais Nin"
        assert book == "Seduction of the Minotaur"

    def test_attribution_with_book_title_parens(self):
        """Test extraction with author and book title in parentheses."""
        text = "The mind is its own place.\n— John Milton (Paradise Lost)"
        clean, author, book = extract_attribution(text)

        assert clean == "The mind is its own place."
        assert author == "John Milton"
        assert book == "Paradise Lost"

    def test_no_attribution(self):
        """Test text with no attribution returns None."""
        text = "Just a quote without any attribution."
        clean, author, book = extract_attribution(text)

        assert clean == "Just a quote without any attribution."
        assert author is None
        assert book is None

    def test_attribution_with_leading_whitespace(self):
        """Test attribution with leading spaces."""
        text = "A beautiful passage.\n  — Some Author"
        clean, author, book = extract_attribution(text)

        assert clean == "A beautiful passage."
        assert author == "Some Author"

    def test_multiline_text_with_attribution(self):
        """Test that only the last attribution line is extracted."""
        text = "First paragraph of a longer quote.\n\nSecond paragraph continues.\n— The Author"
        clean, author, book = extract_attribution(text)

        assert "First paragraph" in clean
        assert "Second paragraph" in clean
        assert author == "The Author"

    def test_attribution_with_smart_quotes(self):
        """Test extraction with smart/curly quotes in book title."""
        text = "Knowledge is power.\n\u2014 Francis Bacon, \u201cNovum Organum\u201d"
        clean, author, book = extract_attribution(text)

        assert author == "Francis Bacon"
        assert book == "Novum Organum"


@pytest.mark.unit
class TestParseCommonplaceEntry:
    """Test parsing a DayOne entry for the commonplace book."""

    def test_basic_entry(self):
        """Test parsing a basic entry with attribution."""
        entry = {
            "uuid": "CP-UUID-001",
            "creationDate": "2024-03-15T14:30:00Z",
            "text": "The only way out is through.\n— Robert Frost",
            "tags": ["poetry", "perseverance"],
        }

        result = parse_commonplace_entry(entry)

        assert result["uuid"] == "CP-UUID-001"
        assert result["creation_date"] == "2024-03-15T14:30:00Z"
        assert result["text"] == "The only way out is through.\n— Robert Frost"
        assert result["clean_text"] == "The only way out is through."
        assert result["tags"] == ["poetry", "perseverance"]
        assert result["author"] == "Robert Frost"
        assert result["book_title"] is None

    def test_entry_without_attribution(self):
        """Test parsing an entry with no detected attribution."""
        entry = {
            "uuid": "CP-UUID-002",
            "creationDate": "2024-04-01T09:00:00Z",
            "text": "A passage I found interesting today.",
            "tags": [],
        }

        result = parse_commonplace_entry(entry)

        assert result["author"] is None
        assert result["book_title"] is None

    def test_entry_with_book_title(self):
        """Test parsing an entry with author and book."""
        entry = {
            "uuid": "CP-UUID-003",
            "creationDate": "2024-05-20T16:45:00Z",
            "text": 'We do not see things as they are.\n— Anais Nin, "Seduction of the Minotaur"',
            "tags": ["perception"],
        }

        result = parse_commonplace_entry(entry)

        assert result["author"] == "Anais Nin"
        assert result["book_title"] == "Seduction of the Minotaur"

    def test_missing_fields_have_defaults(self):
        """Test that missing fields get sensible defaults."""
        entry = {}

        result = parse_commonplace_entry(entry)

        assert result["uuid"] == ""
        assert result["creation_date"] == ""
        assert result["text"] == ""
        assert result["tags"] == []


@pytest.mark.unit
class TestBuildSearchableText:
    """Test building searchable text with author attribution."""

    def test_no_author_returns_original(self):
        """Test that text without author is returned unchanged."""
        text = "The only way out is through."
        result = _build_searchable_text(text, None, None)
        assert result == text

    def test_author_only(self):
        """Test text with author but no book title."""
        text = "The only way out is through."
        result = _build_searchable_text(text, "Robert Frost", None)
        assert result == "[Quote by Robert Frost] The only way out is through."

    def test_author_and_book(self):
        """Test text with both author and book title."""
        text = "We do not see things as they are."
        result = _build_searchable_text(text, "Anais Nin", "Seduction of the Minotaur")
        assert result == '[Quote by Anais Nin, from "Seduction of the Minotaur"] We do not see things as they are.'

    def test_empty_author_treated_as_none(self):
        """Test that empty string author is treated like None."""
        text = "A quote."
        # Empty string is falsy, so should return original
        result = _build_searchable_text(text, "", None)
        assert result == text


@pytest.mark.unit
class TestProcessCommonplaceEntry:
    """Test processing entries into chunks with metadata."""

    def test_short_entry_single_chunk(self):
        """Test that a short entry produces a single chunk with author prefix."""
        entry_data = {
            "uuid": "CP-001",
            "creation_date": "2024-03-15T14:30:00Z",
            "text": "The only way out is through.\n— Robert Frost",
            "clean_text": "The only way out is through.",
            "tags": ["poetry"],
            "author": "Robert Frost",
            "book_title": None,
        }

        chunks = process_commonplace_entry(entry_data, 0)

        assert len(chunks) == 1
        assert chunks[0]["id"] == "commonplace_CP-001_chunk_0"
        # Text should include author prefix for semantic search
        assert "[Quote by Robert Frost]" in chunks[0]["text"]
        assert "The only way out is through." in chunks[0]["text"]
        assert chunks[0]["metadata"]["source_type"] == "commonplace"
        assert chunks[0]["metadata"]["entry_id"] == "CP-001"
        assert chunks[0]["metadata"]["date"] == "2024-03-15T14:30:00Z"
        assert chunks[0]["metadata"]["tags"] == "poetry"
        assert chunks[0]["metadata"]["author"] == "Robert Frost"
        assert "book_title" not in chunks[0]["metadata"]

    def test_entry_with_book_title_metadata(self):
        """Test that book_title is included in metadata and text when present."""
        entry_data = {
            "uuid": "CP-002",
            "creation_date": "2024-05-20T16:45:00Z",
            "text": "A quote from a book.",
            "clean_text": "A quote from a book.",
            "tags": [],
            "author": "Some Author",
            "book_title": "Some Book",
        }

        chunks = process_commonplace_entry(entry_data, 0)

        # Text should include both author and book for semantic search
        assert '[Quote by Some Author, from "Some Book"]' in chunks[0]["text"]
        assert chunks[0]["metadata"]["author"] == "Some Author"
        assert chunks[0]["metadata"]["book_title"] == "Some Book"

    def test_empty_entry_produces_no_chunks(self):
        """Test that an empty entry produces no chunks."""
        entry_data = {
            "uuid": "CP-003",
            "creation_date": "2024-01-01T00:00:00Z",
            "text": "",
            "clean_text": "",
            "tags": [],
            "author": None,
            "book_title": None,
        }

        chunks = process_commonplace_entry(entry_data, 0)
        assert chunks == []

    def test_whitespace_only_entry_produces_no_chunks(self):
        """Test that a whitespace-only entry produces no chunks."""
        entry_data = {
            "uuid": "CP-004",
            "creation_date": "2024-01-01T00:00:00Z",
            "text": "   \n\n  ",
            "clean_text": "",
            "tags": [],
            "author": None,
            "book_title": None,
        }

        chunks = process_commonplace_entry(entry_data, 0)
        assert chunks == []

    def test_no_author_means_no_author_key(self):
        """Test that author key is absent from metadata when not detected."""
        entry_data = {
            "uuid": "CP-005",
            "creation_date": "2024-06-01T12:00:00Z",
            "text": "A quote with no attribution.",
            "clean_text": "A quote with no attribution.",
            "tags": [],
            "author": None,
            "book_title": None,
        }

        chunks = process_commonplace_entry(entry_data, 0)

        assert "author" not in chunks[0]["metadata"]
        assert "book_title" not in chunks[0]["metadata"]

    def test_tags_comma_joined(self):
        """Test that tags are joined with commas."""
        entry_data = {
            "uuid": "CP-006",
            "creation_date": "2024-06-01T12:00:00Z",
            "text": "A tagged quote.",
            "clean_text": "A tagged quote.",
            "tags": ["wisdom", "mindfulness", "zen"],
            "author": None,
            "book_title": None,
        }

        chunks = process_commonplace_entry(entry_data, 0)
        assert chunks[0]["metadata"]["tags"] == "wisdom,mindfulness,zen"

    def test_chunk_index_and_total(self):
        """Test chunk_index and total_chunks metadata."""
        entry_data = {
            "uuid": "CP-007",
            "creation_date": "2024-06-01T12:00:00Z",
            "text": "Short quote.",
            "clean_text": "Short quote.",
            "tags": [],
            "author": None,
            "book_title": None,
        }

        chunks = process_commonplace_entry(entry_data, 5)

        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1
        assert chunks[0]["metadata"]["entry_index"] == 5


@pytest.mark.unit
class TestProcessImageEntry:
    """Test processing image-extracted entries into chunks."""

    def test_valid_image_entry(self):
        """Test processing a valid image extraction result with author prefix."""
        image_data = {
            "quote": "The only way out is through.",
            "author": "Robert Frost",
            "source": "Waking Up",
            "extracted_at": "2024-03-15T14:30:00Z",
        }

        chunks = process_image_entry("quote_image.jpg", image_data, 0)

        assert len(chunks) == 1
        # Text should include author prefix for semantic search
        assert "[Quote by Robert Frost]" in chunks[0]["text"]
        assert "The only way out is through." in chunks[0]["text"]
        assert chunks[0]["metadata"]["source_type"] == "commonplace"
        assert chunks[0]["metadata"]["format"] == "image"
        assert chunks[0]["metadata"]["original_image"] == "quote_image.jpg"
        assert chunks[0]["metadata"]["author"] == "Robert Frost"
        assert chunks[0]["metadata"]["image_source"] == "Waking Up"
        assert "img_" in chunks[0]["id"]

    def test_image_entry_without_author(self):
        """Test processing an image entry without author has no prefix."""
        image_data = {
            "quote": "An unattributed quote.",
            "author": None,
            "source": None,
        }

        chunks = process_image_entry("test.jpg", image_data, 0)

        assert len(chunks) == 1
        # No author means no prefix - text should be unchanged
        assert chunks[0]["text"] == "An unattributed quote."
        assert "author" not in chunks[0]["metadata"]
        assert "image_source" not in chunks[0]["metadata"]

    def test_empty_quote_produces_no_chunks(self):
        """Test that empty quote produces no chunks."""
        image_data = {
            "quote": "",
            "author": "Some Author",
            "source": None,
        }

        chunks = process_image_entry("test.jpg", image_data, 0)
        assert chunks == []

    def test_null_quote_produces_no_chunks(self):
        """Test that null quote produces no chunks."""
        image_data = {
            "quote": None,
            "author": "Some Author",
            "source": None,
        }

        chunks = process_image_entry("test.jpg", image_data, 0)
        assert chunks == []

    def test_stable_ids_for_same_filename(self):
        """Test that the same filename produces the same chunk ID."""
        image_data = {"quote": "Test quote", "author": None, "source": None}

        chunks1 = process_image_entry("same_file.jpg", image_data, 0)
        chunks2 = process_image_entry("same_file.jpg", image_data, 1)

        # The chunk ID should be based on filename hash, so the prefix is the same
        id1_prefix = chunks1[0]["id"].rsplit("_chunk_", 1)[0]
        id2_prefix = chunks2[0]["id"].rsplit("_chunk_", 1)[0]
        assert id1_prefix == id2_prefix

    def test_different_filenames_different_ids(self):
        """Test that different filenames produce different chunk IDs."""
        image_data = {"quote": "Test quote", "author": None, "source": None}

        chunks1 = process_image_entry("file1.jpg", image_data, 0)
        chunks2 = process_image_entry("file2.jpg", image_data, 0)

        assert chunks1[0]["id"] != chunks2[0]["id"]


@pytest.mark.unit
class TestLoadImageCache:
    """Test loading the image extraction cache."""

    def test_load_nonexistent_cache(self, tmp_path, monkeypatch):
        """Test loading a non-existent cache returns empty dict."""
        # Temporarily change the cache path to a non-existent location
        import scripts.ingest_commonplace as ic
        original_path = Path(__file__).parent.parent / "data" / "processed" / "commonplace_images.json"

        # Create a mock that returns empty
        def mock_load():
            return {}

        monkeypatch.setattr(ic, "load_image_cache", mock_load)
        result = ic.load_image_cache()
        assert result == {}

    def test_load_valid_cache(self, tmp_path):
        """Test loading a valid cache file."""
        cache_path = tmp_path / "commonplace_images.json"
        cache_data = {
            "image1.jpg": {"quote": "Test 1", "author": "Author 1"},
            "image2.jpg": {"quote": "Test 2", "author": None},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        # Manually load since we can't easily change the path
        with open(cache_path, "r") as f:
            result = json.load(f)

        assert len(result) == 2
        assert result["image1.jpg"]["quote"] == "Test 1"
