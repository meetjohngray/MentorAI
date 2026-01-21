"""
Retrieval orchestration service.
Combines vector store search with context formatting for RAG.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.config import settings
from app.services.embeddings import get_embedding_service
from app.database.vector_store import get_vector_store, initialize_db

logger = logging.getLogger(__name__)


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
    def is_personal(self) -> bool:
        """Check if this chunk is from personal sources (journals, blog)."""
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
    personal_chunks: List[RetrievedChunk]
    wisdom_chunks: List[RetrievedChunk]


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

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's question or query
            top_k: Number of chunks to retrieve (overrides default)
            source_filter: Optional filter by source type

        Returns:
            RetrievalResult with chunks and formatted context
        """
        top_k = top_k or self.top_k

        # Get vector store
        vector_store = get_vector_store()
        if not vector_store:
            logger.info("Initializing vector store for retrieval...")
            vector_store = initialize_db(settings.chroma_path)

        # Build where filter if specified
        where_filter = None
        if source_filter:
            where_filter = {"source_type": source_filter.lower()}

        # Embed the query
        query_embedding = self.embedding_service.embed_text(query)

        # Search the vector store
        results = vector_store.search(
            query_embedding,
            n_results=top_k,
            where=where_filter
        )

        # Convert to RetrievedChunk objects
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

        # Separate personal and wisdom chunks
        personal_chunks = [c for c in chunks if c.is_personal]
        wisdom_chunks = [c for c in chunks if c.is_wisdom]

        # Format the context
        formatted_context = self._format_context(personal_chunks, wisdom_chunks)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            formatted_context=formatted_context,
            personal_chunks=personal_chunks,
            wisdom_chunks=wisdom_chunks
        )

    def _format_context(
        self,
        personal_chunks: List[RetrievedChunk],
        wisdom_chunks: List[RetrievedChunk]
    ) -> str:
        """
        Format retrieved chunks into context for the prompt.

        Args:
            personal_chunks: Chunks from personal sources
            wisdom_chunks: Chunks from wisdom texts

        Returns:
            Formatted context string
        """
        sections = []

        # Format personal history section
        if personal_chunks:
            personal_section = self._format_personal_chunks(personal_chunks)
            sections.append(personal_section)

        # Format wisdom section
        if wisdom_chunks:
            wisdom_section = self._format_wisdom_chunks(wisdom_chunks)
            sections.append(wisdom_section)

        if not sections:
            return "[No relevant context found]"

        return "\n\n".join(sections)

    def _format_personal_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format personal history chunks."""
        lines = ["=== FROM THE USER'S PERSONAL HISTORY ===\n"]

        for i, chunk in enumerate(chunks, 1):
            # Get date and source info
            date = chunk.metadata.get("date", "Unknown date")
            source = chunk.source_type

            # Format based on source type
            if source == "dayone":
                header = f"[Journal Entry - {date}]"
            elif source == "wordpress":
                title = chunk.metadata.get("title", "Untitled")
                header = f"[Blog Post: \"{title}\" - {date}]"
            else:
                header = f"[{source.title()} - {date}]"

            lines.append(f"--- Entry {i} {header} ---")
            lines.append(chunk.text.strip())
            lines.append("")

        return "\n".join(lines)

    def _format_wisdom_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format wisdom/contemplative text chunks."""
        lines = ["=== FROM CONTEMPLATIVE TRADITIONS ===\n"]

        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown source")
            tradition = chunk.metadata.get("tradition", "")

            if tradition:
                header = f"[{tradition}: {source}]"
            else:
                header = f"[{source}]"

            lines.append(f"--- Wisdom {i} {header} ---")
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
