"""
ChromaDB vector store for personal knowledge retrieval.
Manages embedding storage and similarity search.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store using ChromaDB for semantic search."""

    def __init__(self, persist_directory: str, collection_name: str = "personal_knowledge"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Personal knowledge from journals and other sources"}
        )

        logger.info(f"Collection '{collection_name}' initialized with {self.collection.count()} documents")

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: Unique IDs for each document
            documents: Text content of documents
            embeddings: Embedding vectors for documents
            metadatas: Optional metadata for each document
        """
        logger.info(f"Adding {len(documents)} documents to vector store")

        # ChromaDB 1.4+ requires non-empty metadata dicts
        if not metadatas:
            metadatas = [{"_default": "true"} for _ in range(len(documents))]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Successfully added {len(documents)} documents")

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents using embedding similarity.

        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            where: Optional metadata filters (e.g., {"source_type": "dayone"})

        Returns:
            Dictionary containing ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        # Flatten results (query returns list of lists, we want single result)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": str(self.persist_directory)
        }

    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution)."""
        logger.warning(f"Deleting collection '{self.collection_name}'")
        self.client.delete_collection(name=self.collection_name)
        logger.info("Collection deleted")

    def reset(self) -> None:
        """Reset the database (delete all collections)."""
        logger.warning("Resetting entire database")
        self.client.reset()
        logger.info("Database reset complete")


# Global instance
_vector_store: Optional[VectorStore] = None


def initialize_db(persist_directory: str, collection_name: str = "personal_knowledge") -> VectorStore:
    """
    Initialize the global vector store instance.

    Args:
        persist_directory: Directory path for persistent storage
        collection_name: Name of the ChromaDB collection

    Returns:
        VectorStore instance
    """
    global _vector_store
    _vector_store = VectorStore(persist_directory, collection_name)
    return _vector_store


def get_vector_store() -> Optional[VectorStore]:
    """
    Get the global vector store instance.

    Returns:
        VectorStore instance or None if not initialized
    """
    return _vector_store
