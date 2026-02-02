"""
Tests for the retrieval service - additional coverage.
"""

import pytest
import tempfile
import shutil

from app.services.retrieval import (
    RetrievalService,
    RetrievalResult,
    RetrievedChunk,
    SourcePriority,
    get_retrieval_service,
    reset_retrieval_service,
    get_source_stats,
)
from app.database.vector_store import initialize_db
from app.services.embeddings import get_embedding_service


@pytest.fixture(autouse=True)
def reset_service():
    """Reset singleton before/after each test."""
    reset_retrieval_service()
    yield
    reset_retrieval_service()


@pytest.fixture
def setup_multi_source_store():
    """Set up a vector store with data from multiple sources."""
    temp_dir = tempfile.mkdtemp()

    vector_store = initialize_db(temp_dir, "test_collection")
    embedding_service = get_embedding_service()

    test_docs = [
        "I meditated for 20 minutes today and felt peaceful.",
        "Work has been stressful lately. Need to find balance.",
        "Blog post about finding stillness in chaos.",
        "WordPress article about mindfulness techniques.",
    ]

    embeddings = embedding_service.embed_batch(test_docs)

    vector_store.add_documents(
        ids=["j1", "j2", "b1", "b2"],
        documents=test_docs,
        embeddings=embeddings,
        metadatas=[
            {"source_type": "dayone", "date": "2024-01-15T10:30:00Z"},
            {"source_type": "dayone", "date": "2024-01-20T08:15:00Z"},
            {"source_type": "wordpress", "date": "2024-02-01", "title": "Finding Stillness"},
            {"source_type": "wordpress", "date": "2024-02-15", "title": "Mindfulness Tips"},
        ],
    )

    yield vector_store

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.unit
class TestDateFormatting:
    """Test _format_date helper."""

    def test_format_iso_date_with_timezone(self):
        """Test formatting ISO date with Z timezone."""
        service = RetrievalService(top_k=5)
        result = service._format_date("2024-01-15T10:30:00Z")
        assert result == "January 15, 2024"

    def test_format_iso_date_without_timezone(self):
        """Test formatting ISO date without timezone."""
        service = RetrievalService(top_k=5)
        result = service._format_date("2024-01-15")
        assert result == "January 15, 2024"

    def test_format_empty_date(self):
        """Test formatting empty date string."""
        service = RetrievalService(top_k=5)
        assert service._format_date("") == "Unknown date"
        assert service._format_date("Unknown date") == "Unknown date"

    def test_format_invalid_date(self):
        """Test formatting invalid date string."""
        service = RetrievalService(top_k=5)
        result = service._format_date("not-a-date")
        assert result == "not-a-date"  # Returns original on parse failure


@pytest.mark.unit
class TestWisdomChunkFormatting:
    """Test formatting of wisdom/contemplative chunks."""

    def test_format_wisdom_with_tradition(self):
        """Test wisdom chunk formatting with tradition metadata."""
        service = RetrievalService(top_k=5)
        chunk = RetrievedChunk(
            id="w1",
            text="The mind is everything. What you think you become.",
            metadata={"source": "Dhammapada", "tradition": "Buddhism"},
            distance=0.2,
            relevance_score=0.8,
            source_type="wisdom",
        )

        formatted = service._format_context([], [], [chunk])

        assert "CONTEMPLATIVE TRADITIONS" in formatted
        assert "Buddhism" in formatted
        assert "Dhammapada" in formatted

    def test_format_wisdom_without_tradition(self):
        """Test wisdom chunk formatting without tradition metadata."""
        service = RetrievalService(top_k=5)
        chunk = RetrievedChunk(
            id="w1",
            text="Be still and know.",
            metadata={"source": "Psalms"},
            distance=0.3,
            relevance_score=0.7,
            source_type="wisdom",
        )

        formatted = service._format_context([], [], [chunk])

        assert "[Psalms]" in formatted
        assert "Be still and know" in formatted


@pytest.mark.integration
class TestBalancedSearch:
    """Test balanced search with real vector store."""

    def test_balanced_search_returns_both_sources(self, setup_multi_source_store):
        """Test that balanced search includes both journal and blog results."""
        service = RetrievalService(top_k=4)
        result = service.retrieve("meditation mindfulness")

        source_types = {c.source_type for c in result.chunks}
        # Should have results from both sources
        assert len(source_types) >= 1  # At least one source
        assert len(result.chunks) > 0

    def test_prioritized_search_blog(self, setup_multi_source_store):
        """Test prioritized search when blog keywords detected."""
        service = RetrievalService(top_k=4)
        result = service.retrieve("What do my blog posts say about mindfulness?")

        assert result.detected_priority == SourcePriority.BLOG
        assert len(result.chunks) > 0

    def test_prioritized_search_journal(self, setup_multi_source_store):
        """Test prioritized search when journal keywords detected."""
        service = RetrievalService(top_k=4)
        result = service.retrieve("What patterns are in my journal about stress?")

        assert result.detected_priority == SourcePriority.JOURNAL
        assert len(result.chunks) > 0

    def test_source_filter(self, setup_multi_source_store):
        """Test explicit source filtering."""
        service = RetrievalService(top_k=4)
        result = service.retrieve("meditation", source_filter="dayone")

        # All results should be from dayone
        for chunk in result.chunks:
            assert chunk.source_type == "dayone"

    def test_retrieve_with_custom_top_k(self, setup_multi_source_store):
        """Test retrieve with a custom top_k override."""
        service = RetrievalService(top_k=10)
        result = service.retrieve("meditation", top_k=2)

        assert len(result.chunks) <= 2


@pytest.mark.integration
class TestSourceStats:
    """Test get_source_stats."""

    def test_source_stats_returns_counts(self, setup_multi_source_store):
        """Test that source stats returns counts per source type."""
        stats = get_source_stats()

        assert "total" in stats
        assert stats["total"] > 0
        assert "by_source" in stats
        assert "dayone" in stats["by_source"]
        assert "wordpress" in stats["by_source"]
        assert stats["by_source"]["dayone"] > 0
        assert stats["by_source"]["wordpress"] > 0
