"""
Unit tests for the ChromaDB vector store.
"""

import pytest
from pathlib import Path
from app.database.vector_store import VectorStore, initialize_db, get_vector_store


@pytest.mark.unit
class TestVectorStore:
    """Test the VectorStore class."""

    def test_initialization(self, temp_dir):
        """Test that the vector store initializes correctly."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")

        assert store.collection is not None
        assert store.collection_name == "test_collection"
        assert (temp_dir / "chroma").exists()

    def test_add_documents(self, temp_dir):
        """Test adding documents to the vector store."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")

        ids = ["doc1", "doc2"]
        documents = ["First document", "Second document"]
        # Simple fake embeddings (384 dimensions for all-MiniLM-L6-v2)
        embeddings = [[0.1] * 384, [0.2] * 384]
        metadatas = [
            {"source": "test", "index": 0},
            {"source": "test", "index": 1}
        ]

        store.add_documents(ids, documents, embeddings, metadatas)

        stats = store.get_collection_stats()
        assert stats["total_documents"] == 2

    def test_add_documents_without_metadata(self, temp_dir):
        """Test adding documents without metadata."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")

        ids = ["doc1"]
        documents = ["Test document"]
        embeddings = [[0.1] * 384]

        store.add_documents(ids, documents, embeddings)

        stats = store.get_collection_stats()
        assert stats["total_documents"] == 1

    def test_search(self, temp_dir):
        """Test searching the vector store."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")

        # Add some documents
        ids = ["doc1", "doc2", "doc3"]
        documents = ["meditation", "coding", "nature"]
        embeddings = [
            [1.0] + [0.0] * 383,  # First dimension high
            [0.0] + [1.0] + [0.0] * 382,  # Second dimension high
            [0.0] * 2 + [1.0] + [0.0] * 381  # Third dimension high
        ]
        metadatas = [{"topic": "meditation"}, {"topic": "coding"}, {"topic": "nature"}]

        store.add_documents(ids, documents, embeddings, metadatas)

        # Search with a query embedding similar to the first document
        query_embedding = [0.9] + [0.0] * 383
        results = store.search(query_embedding, n_results=2)

        assert len(results["ids"]) == 2
        assert len(results["documents"]) == 2
        assert len(results["metadatas"]) == 2
        assert len(results["distances"]) == 2
        # The first document should be closest
        assert results["ids"][0] == "doc1"

    def test_search_with_filter(self, temp_dir):
        """Test searching with metadata filters."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")

        ids = ["doc1", "doc2", "doc3"]
        documents = ["meditation", "coding", "nature"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        metadatas = [
            {"source_type": "dayone", "tags": "meditation"},
            {"source_type": "dayone", "tags": "coding"},
            {"source_type": "kindle", "tags": "reading"}
        ]

        store.add_documents(ids, documents, embeddings, metadatas)

        # Search only in dayone entries
        query_embedding = [0.1] * 384
        results = store.search(
            query_embedding,
            n_results=10,
            where={"source_type": "dayone"}
        )

        assert len(results["ids"]) == 2
        assert all(meta["source_type"] == "dayone" for meta in results["metadatas"])

    def test_get_collection_stats(self, temp_dir):
        """Test getting collection statistics."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")

        stats = store.get_collection_stats()
        assert stats["collection_name"] == "test_collection"
        assert stats["total_documents"] == 0
        assert "persist_directory" in stats

        # Add a document and check again
        store.add_documents(["doc1"], ["test"], [[0.1] * 384])

        stats = store.get_collection_stats()
        assert stats["total_documents"] == 1

    def test_persistence(self, temp_dir):
        """Test that data persists across instances."""
        persist_path = str(temp_dir / "chroma")

        # Create first instance and add data
        store1 = VectorStore(persist_path, "test_collection")
        store1.add_documents(["doc1"], ["test document"], [[0.1] * 384])

        # Create second instance and verify data exists
        store2 = VectorStore(persist_path, "test_collection")
        stats = store2.get_collection_stats()
        assert stats["total_documents"] == 1

    def test_delete_collection(self, temp_dir):
        """Test deleting a collection."""
        store = VectorStore(str(temp_dir / "chroma"), "test_collection")
        store.add_documents(["doc1"], ["test"], [[0.1] * 384])

        stats_before = store.get_collection_stats()
        assert stats_before["total_documents"] == 1

        store.delete_collection()

        # Create new instance - should be empty
        store2 = VectorStore(str(temp_dir / "chroma"), "test_collection")
        stats_after = store2.get_collection_stats()
        assert stats_after["total_documents"] == 0


@pytest.mark.unit
class TestVectorStoreGlobalFunctions:
    """Test the global vector store functions."""

    def test_initialize_db(self, temp_dir):
        """Test initializing the global vector store."""
        import app.database.vector_store as vs_module
        vs_module._vector_store = None

        store = initialize_db(str(temp_dir / "chroma"), "test_collection")

        assert store is not None
        assert isinstance(store, VectorStore)

    def test_get_vector_store(self, temp_dir):
        """Test getting the global vector store."""
        import app.database.vector_store as vs_module
        vs_module._vector_store = None

        # Should return None if not initialized
        assert get_vector_store() is None

        # Initialize and then get
        initialize_db(str(temp_dir / "chroma"), "test_collection")
        store = get_vector_store()

        assert store is not None
        assert isinstance(store, VectorStore)

    def test_singleton_behavior(self, temp_dir):
        """Test that initialize_db creates a singleton."""
        import app.database.vector_store as vs_module
        vs_module._vector_store = None

        store1 = initialize_db(str(temp_dir / "chroma"), "test_collection")
        store2 = get_vector_store()

        assert store1 is store2
