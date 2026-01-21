"""
Retrieval orchestration service.
Combines vector store search with context formatting for RAG.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.config import settings
from app.services.embeddings import get_embedding_service
from app.database.vector_store import get_vector_store, initialize_db

logger = logging.getLogger(__name__)


class SourcePriority(Enum):
    """Priority for source types based on query analysis."""
    NONE = "none"  # No specific source priority
    JOURNAL = "journal"  # Prioritize DayOne/private writing
    BLOG = "blog"  # Prioritize WordPress/public writing


# Keywords that suggest the user is asking about specific source types
BLOG_KEYWORDS = [
    r"\bblog\b", r"\bpost\b", r"\bposts\b", r"\barticle\b", r"\barticles\b",
    r"\bpublic writing\b", r"\bpublished\b", r"\bessay\b", r"\bessays\b",
    r"\bwordpress\b", r"\bpublic\b"
]
JOURNAL_KEYWORDS = [
    r"\bjournal\b", r"\bdiary\b", r"\bprivate\b", r"\bpersonal\b",
    r"\bentry\b", r"\bentries\b", r"\bdayone\b", r"\bday one\b",
    r"\bprivate writing\b", r"\breflection\b", r"\breflections\b"
]


@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from the vector store."""
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    relevance_score: float
    source_type: str

    @property
    def is_journal(self) -> bool:
        """Check if this chunk is from private journal (DayOne)."""
        return self.source_type == "dayone"

    @property
    def is_blog(self) -> bool:
        """Check if this chunk is from public blog (WordPress)."""
        return self.source_type == "wordpress"

    @property
    def is_personal(self) -> bool:
        """Check if this chunk is from user's own writing (journal or blog)."""
        return self.source_type in ("dayone", "wordpress")

    @property
    def is_wisdom(self) -> bool:
        """Check if this chunk is from wisdom/contemplative texts."""
        return self.source_type == "wisdom"


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: str
    chunks: List[RetrievedChunk]
    formatted_context: str
    journal_chunks: List[RetrievedChunk]
    blog_chunks: List[RetrievedChunk]
    wisdom_chunks: List[RetrievedChunk]
    detected_priority: SourcePriority

    @property
    def personal_chunks(self) -> List[RetrievedChunk]:
        """All personal writing chunks (journal + blog)."""
        return self.journal_chunks + self.blog_chunks


class RetrievalService:
    """Service for retrieving and formatting context for RAG."""

    def __init__(self, top_k: int = 10):
        """
        Initialize the retrieval service.

        Args:
            top_k: Number of chunks to retrieve
        """
        self.top_k = top_k
        self.embedding_service = get_embedding_service(settings.embedding_model)

    def _detect_source_priority(self, query: str) -> SourcePriority:
        """
        Analyze the query to detect if user is asking about a specific source type.

        Args:
            query: The user's question

        Returns:
            SourcePriority indicating which source to prioritize
        """
        query_lower = query.lower()

        # Check for blog/public writing keywords
        blog_matches = sum(1 for pattern in BLOG_KEYWORDS if re.search(pattern, query_lower))
        journal_matches = sum(1 for pattern in JOURNAL_KEYWORDS if re.search(pattern, query_lower))

        if blog_matches > journal_matches and blog_matches > 0:
            return SourcePriority.BLOG
        elif journal_matches > blog_matches and journal_matches > 0:
            return SourcePriority.JOURNAL
        return SourcePriority.NONE

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Searches ALL sources by default, but can prioritize specific sources
        based on query analysis or explicit filter.

        Args:
            query: The user's question or query
            top_k: Number of chunks to retrieve (overrides default)
            source_filter: Optional explicit filter by source type

        Returns:
            RetrievalResult with chunks and formatted context
        """
        top_k = top_k or self.top_k

        # Get vector store
        vector_store = get_vector_store()
        if not vector_store:
            logger.info("Initializing vector store for retrieval...")
            vector_store = initialize_db(settings.chroma_path)

        # Detect what source type the user might be asking about
        detected_priority = self._detect_source_priority(query)
        logger.info(f"Detected source priority: {detected_priority.value}")

        # Build where filter if explicitly specified
        where_filter = None
        if source_filter:
            where_filter = {"source_type": source_filter.lower()}

        # Embed the query
        query_embedding = self.embedding_service.embed_text(query)

        # Search strategy based on detected priority and filters
        if source_filter:
            # Explicit filter - search only that source
            results = vector_store.search(
                query_embedding,
                n_results=top_k,
                where=where_filter
            )
            chunks = self._results_to_chunks(results)
        elif detected_priority == SourcePriority.BLOG:
            # User asking about blog - 80% blog, 20% journal
            chunks = self._prioritized_search(
                vector_store, query_embedding, top_k,
                primary_source="wordpress", primary_ratio=0.8
            )
        elif detected_priority == SourcePriority.JOURNAL:
            # User asking about journal - 80% journal, 20% blog
            chunks = self._prioritized_search(
                vector_store, query_embedding, top_k,
                primary_source="dayone", primary_ratio=0.8
            )
        else:
            # General query - balanced 50/50 retrieval from each source
            chunks = self._balanced_search(vector_store, query_embedding, top_k)

        # Separate by source type
        journal_chunks = [c for c in chunks if c.is_journal]
        blog_chunks = [c for c in chunks if c.is_blog]
        wisdom_chunks = [c for c in chunks if c.is_wisdom]

        # Format the context with clear source labels
        formatted_context = self._format_context(journal_chunks, blog_chunks, wisdom_chunks)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            formatted_context=formatted_context,
            journal_chunks=journal_chunks,
            blog_chunks=blog_chunks,
            wisdom_chunks=wisdom_chunks,
            detected_priority=detected_priority
        )

    def _balanced_search(
        self,
        vector_store,
        query_embedding: List[float],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Search with balanced representation from all source types.

        Queries each source separately to ensure representation regardless
        of how much content exists in each source.

        Args:
            vector_store: The vector store to search
            query_embedding: Embedded query
            top_k: Total number of results

        Returns:
            List of RetrievedChunk objects with balanced source representation
        """
        # Split evenly between sources (can add wisdom later)
        per_source = max(1, top_k // 2)

        # Get top results from DayOne (journal)
        journal_results = vector_store.search(
            query_embedding,
            n_results=per_source,
            where={"source_type": "dayone"}
        )
        journal_chunks = self._results_to_chunks(journal_results)

        # Get top results from WordPress (blog)
        blog_results = vector_store.search(
            query_embedding,
            n_results=per_source,
            where={"source_type": "wordpress"}
        )
        blog_chunks = self._results_to_chunks(blog_results)

        # Combine and sort by relevance
        all_chunks = journal_chunks + blog_chunks
        all_chunks.sort(key=lambda c: c.relevance_score, reverse=True)

        logger.info(
            f"Balanced search: {len(journal_chunks)} journal, "
            f"{len(blog_chunks)} blog chunks retrieved"
        )

        return all_chunks[:top_k]

    def _prioritized_search(
        self,
        vector_store,
        query_embedding: List[float],
        top_k: int,
        primary_source: str,
        primary_ratio: float = 0.8
    ) -> List[RetrievedChunk]:
        """
        Search with priority given to a specific source type.

        Args:
            vector_store: The vector store to search
            query_embedding: Embedded query
            top_k: Total number of results
            primary_source: Source type to prioritize (dayone, wordpress)
            primary_ratio: Ratio of results from primary source (0.0-1.0)

        Returns:
            List of RetrievedChunk objects
        """
        primary_count = max(1, int(top_k * primary_ratio))
        secondary_count = max(1, top_k - primary_count)

        # Get results from primary source
        primary_results = vector_store.search(
            query_embedding,
            n_results=primary_count,
            where={"source_type": primary_source}
        )
        primary_chunks = self._results_to_chunks(primary_results)

        # Get results from other sources
        other_source = "dayone" if primary_source == "wordpress" else "wordpress"
        secondary_results = vector_store.search(
            query_embedding,
            n_results=secondary_count,
            where={"source_type": other_source}
        )
        secondary_chunks = self._results_to_chunks(secondary_results)

        # Combine and sort by relevance
        all_chunks = primary_chunks + secondary_chunks
        all_chunks.sort(key=lambda c: c.relevance_score, reverse=True)

        logger.info(
            f"Prioritized search ({primary_source}): "
            f"{len(primary_chunks)} primary, {len(secondary_chunks)} secondary"
        )

        return all_chunks[:top_k]

    def _results_to_chunks(self, results: Dict[str, Any]) -> List[RetrievedChunk]:
        """Convert vector store results to RetrievedChunk objects."""
        chunks = []
        for idx in range(len(results["ids"])):
            metadata = results["metadatas"][idx]
            source_type = metadata.get("source_type", "unknown")

            chunk = RetrievedChunk(
                id=results["ids"][idx],
                text=results["documents"][idx],
                metadata=metadata,
                distance=results["distances"][idx],
                relevance_score=1 - results["distances"][idx],
                source_type=source_type
            )
            chunks.append(chunk)
        return chunks

    def _format_context(
        self,
        journal_chunks: List[RetrievedChunk],
        blog_chunks: List[RetrievedChunk],
        wisdom_chunks: List[RetrievedChunk]
    ) -> str:
        """
        Format retrieved chunks into context for the prompt.

        Args:
            journal_chunks: Chunks from private journal (DayOne)
            blog_chunks: Chunks from public blog (WordPress)
            wisdom_chunks: Chunks from wisdom texts

        Returns:
            Formatted context string with clear source labels
        """
        sections = []

        # Format private journal section
        if journal_chunks:
            journal_section = self._format_journal_chunks(journal_chunks)
            sections.append(journal_section)

        # Format public blog section
        if blog_chunks:
            blog_section = self._format_blog_chunks(blog_chunks)
            sections.append(blog_section)

        # Format wisdom section
        if wisdom_chunks:
            wisdom_section = self._format_wisdom_chunks(wisdom_chunks)
            sections.append(wisdom_section)

        if not sections:
            return "[No relevant context found]"

        return "\n\n".join(sections)

    def _format_date(self, date_str: str) -> str:
        """Format a date string for display."""
        if not date_str or date_str == "Unknown date":
            return "Unknown date"
        # Try to parse and format nicely
        try:
            from datetime import datetime
            # Handle ISO format dates
            if "T" in date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(date_str)
            return dt.strftime("%B %d, %Y")
        except (ValueError, TypeError):
            return date_str

    def _format_journal_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format private journal chunks (DayOne)."""
        lines = ["=== FROM YOUR PRIVATE JOURNAL ==="]
        lines.append("(Personal reflections written for yourself, not for others)\n")

        for chunk in chunks:
            date = self._format_date(chunk.metadata.get("date", "Unknown date"))
            lines.append(f"[From your personal journal - {date}]")
            lines.append(chunk.text.strip())
            lines.append("")

        return "\n".join(lines)

    def _format_blog_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format public blog chunks (WordPress)."""
        lines = ["=== FROM YOUR PUBLIC WRITING ==="]
        lines.append("(Blog posts and essays you published for others to read)\n")

        for chunk in chunks:
            date = self._format_date(chunk.metadata.get("date", "Unknown date"))
            title = chunk.metadata.get("title", "Untitled")
            lines.append(f"[From your blog post \"{title}\" - {date}]")
            lines.append(chunk.text.strip())
            lines.append("")

        return "\n".join(lines)

    def _format_wisdom_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format wisdom/contemplative text chunks."""
        lines = ["=== FROM CONTEMPLATIVE TRADITIONS ===\n"]

        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown source")
            tradition = chunk.metadata.get("tradition", "")

            if tradition:
                header = f"[{tradition}: {source}]"
            else:
                header = f"[{source}]"

            lines.append(header)
            lines.append(chunk.text.strip())
            lines.append("")

        return "\n".join(lines)


# Global instance
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """
    Get the global retrieval service instance (singleton pattern).

    Returns:
        RetrievalService instance
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(top_k=settings.retrieval_top_k)
    return _retrieval_service


def reset_retrieval_service() -> None:
    """Reset the global retrieval service instance (useful for testing)."""
    global _retrieval_service
    _retrieval_service = None


def get_source_stats() -> Dict[str, Any]:
    """
    Get statistics about chunks per source type in the vector store.

    Useful for debugging and understanding data distribution.

    Returns:
        Dictionary with counts per source type and total
    """
    vector_store = get_vector_store()
    if not vector_store:
        vector_store = initialize_db(settings.chroma_path)

    total = vector_store.collection.count()

    # Count by source type using metadata queries
    # Note: ChromaDB doesn't have a direct count-by-filter, so we use a workaround
    stats = {
        "total": total,
        "by_source": {},
    }

    # Get a sample to determine source types present
    if total > 0:
        # Query with a dummy embedding to get metadata
        # We'll use the embedding service to create a neutral query
        embedding_service = get_embedding_service(settings.embedding_model)
        dummy_embedding = embedding_service.embed_text("content")

        for source_type in ["dayone", "wordpress", "wisdom"]:
            try:
                results = vector_store.search(
                    dummy_embedding,
                    n_results=1,
                    where={"source_type": source_type}
                )
                # If we get results, the source type exists
                # For accurate count, we'd need to iterate, but this gives presence
                if results["ids"]:
                    # Get approximate count by querying more
                    full_results = vector_store.search(
                        dummy_embedding,
                        n_results=min(10000, total),
                        where={"source_type": source_type}
                    )
                    stats["by_source"][source_type] = len(full_results["ids"])
                else:
                    stats["by_source"][source_type] = 0
            except Exception as e:
                logger.warning(f"Could not count {source_type}: {e}")
                stats["by_source"][source_type] = "unknown"

    return stats
