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


@pytest.fixture
def setup_multi_source_store_with_wisdom():
    """Set up a vector store with data from all three sources including wisdom."""
    temp_dir = tempfile.mkdtemp()

    vector_store = initialize_db(temp_dir, "test_collection")
    embedding_service = get_embedding_service()

    test_docs = [
        "I meditated for 20 minutes today and felt peaceful.",
        "Work has been stressful lately. Need to find balance.",
        "Blog post about finding stillness in chaos.",
        "WordPress article about mindfulness techniques.",
        "The mind is everything. What you think you become.",
        "In the beginner's mind there are many possibilities.",
    ]

    embeddings = embedding_service.embed_batch(test_docs)

    vector_store.add_documents(
        ids=["j1", "j2", "b1", "b2", "w1", "w2"],
        documents=test_docs,
        embeddings=embeddings,
        metadatas=[
            {"source_type": "dayone", "date": "2024-01-15T10:30:00Z"},
            {"source_type": "dayone", "date": "2024-01-20T08:15:00Z"},
            {"source_type": "wordpress", "date": "2024-02-01", "title": "Finding Stillness"},
            {"source_type": "wordpress", "date": "2024-02-15", "title": "Mindfulness Tips"},
            {"source_type": "wisdom", "source": "Dhammapada", "tradition": "Buddhism"},
            {"source_type": "wisdom", "source": "Shunryu Suzuki", "tradition": "Zen Buddhism"},
        ],
    )

    yield vector_store

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.unit
class TestWisdomKeywordDetection:
    """Test wisdom keyword detection in query analysis."""

    def test_detects_wisdom_keywords(self):
        """Test that wisdom-related queries are detected."""
        service = RetrievalService(top_k=5)

        assert service._detect_source_priority("What do the Buddhist teachings say?") == SourcePriority.WISDOM
        assert service._detect_source_priority("Tell me about zen koans") == SourcePriority.WISDOM
        assert service._detect_source_priority("What does advaita vedanta teach?") == SourcePriority.WISDOM

    def test_wisdom_vs_blog_priority(self):
        """Test that the highest keyword count wins."""
        service = RetrievalService(top_k=5)

        # Blog keyword wins when more blog keywords present
        assert service._detect_source_priority("my blog posts about wisdom") == SourcePriority.BLOG

    def test_no_priority_on_general_query(self):
        """Test that general queries have no priority."""
        service = RetrievalService(top_k=5)
        assert service._detect_source_priority("How can I deal with stress?") == SourcePriority.NONE


@pytest.mark.integration
class TestBalancedSearchWithWisdom:
    """Test balanced search with all three source types."""

    def test_balanced_search_includes_wisdom(self, setup_multi_source_store_with_wisdom):
        """Test that balanced search can include wisdom results."""
        service = RetrievalService(top_k=6)
        result = service.retrieve("mindfulness meditation practice")

        source_types = {c.source_type for c in result.chunks}
        # Should have results from at least 2 sources
        assert len(source_types) >= 2
        assert len(result.chunks) > 0

    def test_wisdom_source_filter(self, setup_multi_source_store_with_wisdom):
        """Test explicit wisdom source filtering."""
        service = RetrievalService(top_k=4)
        result = service.retrieve("meditation", source_filter="wisdom")

        for chunk in result.chunks:
            assert chunk.source_type == "wisdom"


@pytest.mark.integration
class TestWisdomPrioritizedSearch:
    """Test wisdom-prioritized search."""

    def test_wisdom_priority_detected(self, setup_multi_source_store_with_wisdom):
        """Test that wisdom keywords trigger wisdom priority."""
        service = RetrievalService(top_k=6)
        result = service.retrieve("What do the Buddhist teachings say about the mind?")

        assert result.detected_priority == SourcePriority.WISDOM
        assert len(result.chunks) > 0

    def test_wisdom_chunks_in_result(self, setup_multi_source_store_with_wisdom):
        """Test that wisdom chunks appear in wisdom-prioritized results."""
        service = RetrievalService(top_k=6)
        result = service.retrieve("What do zen traditions teach about beginner's mind?")

        assert len(result.wisdom_chunks) > 0


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

    def test_source_stats_includes_wisdom(self, setup_multi_source_store_with_wisdom):
        """Test that source stats includes wisdom counts."""
        stats = get_source_stats()

        assert stats["by_source"]["wisdom"] > 0


@pytest.mark.unit
class TestCommonplaceKeywordDetection:
    """Test commonplace keyword detection in query analysis."""

    def test_detects_commonplace_keywords(self):
        """Test that commonplace-related queries are detected."""
        service = RetrievalService(top_k=5)

        assert service._detect_source_priority("Show me quotes I've saved") == SourcePriority.COMMONPLACE
        assert service._detect_source_priority("What's in my commonplace book?") == SourcePriority.COMMONPLACE
        assert service._detect_source_priority("What passages have I collected?") == SourcePriority.COMMONPLACE

    def test_commonplace_vs_wisdom_priority(self):
        """Test that the highest keyword count wins between commonplace and wisdom."""
        service = RetrievalService(top_k=5)

        # Wisdom should win with more wisdom keywords
        assert service._detect_source_priority("Buddhist teachings about wisdom") == SourcePriority.WISDOM


@pytest.mark.unit
class TestCommonplaceChunkFormatting:
    """Test formatting of commonplace book chunks."""

    def test_format_commonplace_with_author(self):
        """Test commonplace chunk formatting with author metadata."""
        service = RetrievalService(top_k=5)
        chunk = RetrievedChunk(
            id="cp1",
            text="The only way out is through.",
            metadata={"date": "2024-03-15T14:30:00Z", "author": "Robert Frost"},
            distance=0.2,
            relevance_score=0.8,
            source_type="commonplace",
        )

        formatted = service._format_context([], [], [], [chunk])

        assert "COMMONPLACE BOOK" in formatted
        assert "Robert Frost" in formatted
        assert "The only way out is through." in formatted

    def test_format_commonplace_with_author_and_book(self):
        """Test commonplace chunk formatting with author and book title."""
        service = RetrievalService(top_k=5)
        chunk = RetrievedChunk(
            id="cp1",
            text="We do not see things as they are.",
            metadata={
                "date": "2024-05-20T16:45:00Z",
                "author": "Anais Nin",
                "book_title": "Seduction of the Minotaur",
            },
            distance=0.2,
            relevance_score=0.8,
            source_type="commonplace",
        )

        formatted = service._format_context([], [], [], [chunk])

        assert "Anais Nin" in formatted
        assert "Seduction of the Minotaur" in formatted

    def test_format_commonplace_without_author(self):
        """Test commonplace chunk formatting without author metadata."""
        service = RetrievalService(top_k=5)
        chunk = RetrievedChunk(
            id="cp1",
            text="An unattributed passage.",
            metadata={"date": "2024-06-01T12:00:00Z"},
            distance=0.3,
            relevance_score=0.7,
            source_type="commonplace",
        )

        formatted = service._format_context([], [], [], [chunk])

        assert "COMMONPLACE BOOK" in formatted
        assert "An unattributed passage." in formatted


@pytest.mark.unit
class TestRetrievedChunkProperties:
    """Test RetrievedChunk is_commonplace property."""

    def test_is_commonplace(self):
        """Test is_commonplace property."""
        chunk = RetrievedChunk(
            id="cp1",
            text="A collected quote.",
            metadata={"source_type": "commonplace"},
            distance=0.2,
            relevance_score=0.8,
            source_type="commonplace",
        )
        assert chunk.is_commonplace is True
        assert chunk.is_wisdom is False
        assert chunk.is_personal is False

    def test_is_not_commonplace(self):
        """Test is_commonplace returns False for other source types."""
        chunk = RetrievedChunk(
            id="w1",
            text="A wisdom text.",
            metadata={"source_type": "wisdom"},
            distance=0.2,
            relevance_score=0.8,
            source_type="wisdom",
        )
        assert chunk.is_commonplace is False


@pytest.fixture
def setup_all_sources_store():
    """Set up a vector store with data from all four sources."""
    temp_dir = tempfile.mkdtemp()

    vector_store = initialize_db(temp_dir, "test_collection")
    embedding_service = get_embedding_service()

    test_docs = [
        "I meditated for 20 minutes today and felt peaceful.",
        "Work has been stressful lately. Need to find balance.",
        "Blog post about finding stillness in chaos.",
        "WordPress article about mindfulness techniques.",
        "The mind is everything. What you think you become.",
        "In the beginner's mind there are many possibilities.",
        "The only way out is through.",
        "We do not see things as they are, we see them as we are.",
    ]

    embeddings = embedding_service.embed_batch(test_docs)

    vector_store.add_documents(
        ids=["j1", "j2", "b1", "b2", "w1", "w2", "cp1", "cp2"],
        documents=test_docs,
        embeddings=embeddings,
        metadatas=[
            {"source_type": "dayone", "date": "2024-01-15T10:30:00Z"},
            {"source_type": "dayone", "date": "2024-01-20T08:15:00Z"},
            {"source_type": "wordpress", "date": "2024-02-01", "title": "Finding Stillness"},
            {"source_type": "wordpress", "date": "2024-02-15", "title": "Mindfulness Tips"},
            {"source_type": "wisdom", "source": "Dhammapada", "tradition": "Buddhism"},
            {"source_type": "wisdom", "source": "Shunryu Suzuki", "tradition": "Zen Buddhism"},
            {"source_type": "commonplace", "date": "2024-03-15", "author": "Robert Frost"},
            {"source_type": "commonplace", "date": "2024-05-20", "author": "Anais Nin"},
        ],
    )

    yield vector_store

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
class TestCommonplaceSearch:
    """Test search with commonplace source type."""

    def test_commonplace_source_filter(self, setup_all_sources_store):
        """Test explicit commonplace source filtering."""
        service = RetrievalService(top_k=4)
        result = service.retrieve("quotes", source_filter="commonplace")

        for chunk in result.chunks:
            assert chunk.source_type == "commonplace"

    def test_commonplace_priority_detected(self, setup_all_sources_store):
        """Test that commonplace keywords trigger commonplace priority."""
        service = RetrievalService(top_k=8)
        result = service.retrieve("Show me the quotes I've collected and saved")

        assert result.detected_priority == SourcePriority.COMMONPLACE
        assert len(result.chunks) > 0

    def test_commonplace_chunks_in_result(self, setup_all_sources_store):
        """Test that commonplace_chunks field is populated."""
        service = RetrievalService(top_k=8)
        result = service.retrieve("quotes I've saved and collected")

        assert len(result.commonplace_chunks) > 0

    def test_balanced_search_includes_all_sources(self, setup_all_sources_store):
        """Test that balanced search can include all four source types."""
        service = RetrievalService(top_k=8)
        result = service.retrieve("mindfulness meditation practice")

        source_types = {c.source_type for c in result.chunks}
        # Should have results from at least 2 sources
        assert len(source_types) >= 2
        assert len(result.chunks) > 0

    def test_source_stats_includes_commonplace(self, setup_all_sources_store):
        """Test that source stats includes commonplace counts."""
        stats = get_source_stats()

        assert stats["by_source"]["commonplace"] > 0
