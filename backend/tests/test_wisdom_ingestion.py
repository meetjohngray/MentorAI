"""
Tests for the wisdom text ingestion pipeline.
"""

import json
import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ingest_wisdom import (
    infer_tradition,
    parse_text_title,
    load_sources_metadata,
    process_wisdom_file,
    WISDOM_TARGET_TOKENS,
    WISDOM_MAX_TOKENS,
)


@pytest.mark.unit
class TestInferTradition:
    """Test tradition directory name → display name mapping."""

    def test_known_traditions(self):
        assert infer_tradition("advaita") == "Advaita Vedanta"
        assert infer_tradition("buddhist") == "Buddhism"
        assert infer_tradition("zen") == "Zen Buddhism"
        assert infer_tradition("tao") == "Taoism"
        assert infer_tradition("stoic") == "Stoicism"

    def test_case_insensitive(self):
        assert infer_tradition("Advaita") == "Advaita Vedanta"
        assert infer_tradition("ZEN") == "Zen Buddhism"
        assert infer_tradition("BUDDHIST") == "Buddhism"

    def test_unknown_tradition_title_cased(self):
        assert infer_tradition("mysticism") == "Mysticism"
        assert infer_tradition("some_tradition") == "Some_Tradition"


@pytest.mark.unit
class TestParseTextTitle:
    """Test filename → human-readable title conversion."""

    def test_underscores_to_spaces(self):
        assert parse_text_title("who_am_i.txt") == "Who Am I"
        assert parse_text_title("faith_in_mind.txt") == "Faith In Mind"

    def test_hyphens_to_spaces(self):
        assert parse_text_title("heart-sutra.txt") == "Heart Sutra"

    def test_no_extension(self):
        assert parse_text_title("dhammapada") == "Dhammapada"

    def test_mixed_separators(self):
        assert parse_text_title("grass_roof-hermitage.txt") == "Grass Roof Hermitage"


@pytest.mark.unit
class TestLoadSourcesMetadata:
    """Test loading metadata from sources.json."""

    def test_loads_valid_sources_json(self, tmp_path):
        """Test loading a well-formed sources.json."""
        sources = {
            "traditions": {
                "zen": {
                    "display_name": "Zen Buddhism",
                    "texts": [
                        {
                            "filename": "gateless_gate.txt",
                            "title": "The Gateless Gate",
                            "teacher": "Wumen Huikai",
                            "attribution": "Sacred Texts Archive",
                        }
                    ],
                }
            }
        }
        sources_path = tmp_path / "sources.json"
        sources_path.write_text(json.dumps(sources))

        lookup = load_sources_metadata(tmp_path)

        assert "gateless_gate.txt" in lookup
        assert lookup["gateless_gate.txt"]["title"] == "The Gateless Gate"
        assert lookup["gateless_gate.txt"]["teacher"] == "Wumen Huikai"
        assert lookup["gateless_gate.txt"]["tradition"] == "Zen Buddhism"
        assert lookup["gateless_gate.txt"]["tradition_key"] == "zen"

    def test_returns_empty_when_no_file(self, tmp_path):
        """Test graceful fallback when sources.json doesn't exist."""
        lookup = load_sources_metadata(tmp_path)
        assert lookup == {}

    def test_handles_missing_optional_fields(self, tmp_path):
        """Test that missing teacher/attribution get defaults."""
        sources = {
            "traditions": {
                "buddhist": {
                    "display_name": "Buddhism",
                    "texts": [
                        {
                            "filename": "dhammapada.txt",
                            "title": "The Dhammapada",
                        }
                    ],
                }
            }
        }
        sources_path = tmp_path / "sources.json"
        sources_path.write_text(json.dumps(sources))

        lookup = load_sources_metadata(tmp_path)
        assert lookup["dhammapada.txt"]["teacher"] == "Unknown"
        assert lookup["dhammapada.txt"]["attribution"] == ""


@pytest.mark.unit
class TestProcessWisdomFile:
    """Test processing a single wisdom text file into chunks."""

    def test_short_text_single_chunk(self, tmp_path):
        """Test that a short text produces a single chunk."""
        text_file = tmp_path / "short_text.txt"
        text_file.write_text("This is a short wisdom teaching about presence.")

        chunks = process_wisdom_file(
            text_file,
            tradition="Zen Buddhism",
            tradition_key="zen",
            metadata_lookup={},
            file_index=0,
        )

        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a short wisdom teaching about presence."
        assert chunks[0]["id"] == "wisdom_zen_short_text_chunk_0"
        assert chunks[0]["metadata"]["source_type"] == "wisdom"
        assert chunks[0]["metadata"]["tradition"] == "Zen Buddhism"
        assert chunks[0]["metadata"]["tradition_key"] == "zen"
        assert chunks[0]["metadata"]["text_title"] == "Short Text"
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1

    def test_uses_sources_json_metadata(self, tmp_path):
        """Test that sources.json metadata overrides inferred values."""
        text_file = tmp_path / "gateless_gate.txt"
        text_file.write_text("A monk asked Joshu: Does a dog have Buddha-nature?")

        metadata_lookup = {
            "gateless_gate.txt": {
                "title": "The Gateless Gate",
                "teacher": "Wumen Huikai",
                "tradition": "Zen Buddhism",
                "tradition_key": "zen",
                "attribution": "Sacred Texts Archive",
            }
        }

        chunks = process_wisdom_file(
            text_file,
            tradition="Zen Buddhism",
            tradition_key="zen",
            metadata_lookup=metadata_lookup,
            file_index=0,
        )

        assert chunks[0]["metadata"]["text_title"] == "The Gateless Gate"
        assert chunks[0]["metadata"]["teacher"] == "Wumen Huikai"
        assert chunks[0]["metadata"]["source"] == "The Gateless Gate by Wumen Huikai"
        assert chunks[0]["metadata"]["attribution"] == "Sacred Texts Archive"

    def test_long_text_produces_multiple_chunks(self, tmp_path):
        """Test that a long text is split into multiple chunks."""
        # Create text long enough to exceed WISDOM_MAX_TOKENS (~1000 tokens = ~4000 chars)
        paragraphs = ["This is a paragraph about mindfulness and awareness. " * 20 for _ in range(10)]
        long_text = "\n\n".join(paragraphs)

        text_file = tmp_path / "long_teaching.txt"
        text_file.write_text(long_text)

        chunks = process_wisdom_file(
            text_file,
            tradition="Buddhism",
            tradition_key="buddhist",
            metadata_lookup={},
            file_index=0,
        )

        assert len(chunks) > 1
        # All chunks should have consistent metadata
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["source_type"] == "wisdom"
            assert chunk["metadata"]["tradition"] == "Buddhism"
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)
            assert chunk["id"] == f"wisdom_buddhist_long_teaching_chunk_{i}"

    def test_skips_empty_file(self, tmp_path):
        """Test that an empty file produces no chunks."""
        text_file = tmp_path / "empty.txt"
        text_file.write_text("")

        chunks = process_wisdom_file(
            text_file,
            tradition="General",
            tradition_key="general",
            metadata_lookup={},
            file_index=0,
        )

        assert chunks == []

    def test_unknown_teacher_source_label(self, tmp_path):
        """Test source label when teacher is Unknown."""
        text_file = tmp_path / "anonymous_text.txt"
        text_file.write_text("A teaching of unknown origin.")

        chunks = process_wisdom_file(
            text_file,
            tradition="General",
            tradition_key="general",
            metadata_lookup={},
            file_index=0,
        )

        # When teacher is "Unknown", source should just be the title
        assert chunks[0]["metadata"]["source"] == "Anonymous Text"
        assert chunks[0]["metadata"]["teacher"] == "Unknown"


@pytest.mark.unit
class TestWisdomChunkSizes:
    """Test that wisdom texts use the correct chunk sizes."""

    def test_chunk_sizes_are_larger_than_defaults(self):
        """Wisdom chunk sizes should be larger than the default journal/blog sizes."""
        from app.config import settings

        assert WISDOM_TARGET_TOKENS > settings.chunk_target_tokens
        assert WISDOM_MAX_TOKENS > settings.chunk_max_tokens

    def test_target_and_max_values(self):
        """Verify the specific chunk size values."""
        assert WISDOM_TARGET_TOKENS == 800
        assert WISDOM_MAX_TOKENS == 1000
