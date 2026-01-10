"""
Unit tests for the embeddings service.
"""

import pytest
from app.services.embeddings import EmbeddingService, get_embedding_service


@pytest.mark.unit
class TestEmbeddingService:
    """Test the EmbeddingService class."""

    def test_initialization(self):
        """Test that the service initializes correctly."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        assert service.model is not None
        assert service.model_name == "all-MiniLM-L6-v2"

    def test_embed_single_text(self):
        """Test embedding a single text."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        text = "This is a test sentence about meditation."
        embedding = service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self):
        """Test embedding multiple texts in a batch."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        texts = [
            "First test sentence about meditation.",
            "Second test sentence about coding.",
            "Third test sentence about nature."
        ]
        embeddings = service.embed_batch(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_embedding_dimension(self):
        """Test getting the embedding dimension."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        dimension = service.get_embedding_dimension()

        # all-MiniLM-L6-v2 has 384 dimensions
        assert dimension == 384

    def test_embedding_consistency(self):
        """Test that the same text produces the same embedding."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        text = "Consistent test sentence."

        embedding1 = service.embed_text(text)
        embedding2 = service.embed_text(text)

        # Embeddings should be identical for the same text
        assert embedding1 == embedding2

    def test_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        text1 = "I love meditation and mindfulness."
        text2 = "Meditation and being mindful are great."
        text3 = "Python programming is interesting."

        emb1 = service.embed_text(text1)
        emb2 = service.embed_text(text2)
        emb3 = service.embed_text(text3)

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x * x for x in a) ** 0.5
            mag_b = sum(y * y for y in b) ** 0.5
            return dot_product / (mag_a * mag_b)

        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)

        # Similar texts should be more similar than dissimilar ones
        assert sim_1_2 > sim_1_3

    def test_batch_size_parameter(self):
        """Test that batch_size parameter works."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        texts = [f"Test sentence {i}" for i in range(10)]

        embeddings = service.embed_batch(texts, batch_size=2)
        assert len(embeddings) == 10


@pytest.mark.unit
class TestGetEmbeddingService:
    """Test the singleton pattern for embedding service."""

    def test_singleton_returns_same_instance(self):
        """Test that get_embedding_service returns the same instance."""
        # Reset the global instance for this test
        import app.services.embeddings as emb_module
        emb_module._embedding_service = None

        service1 = get_embedding_service()
        service2 = get_embedding_service()

        assert service1 is service2

    def test_singleton_initialization(self):
        """Test that the singleton is properly initialized."""
        import app.services.embeddings as emb_module
        emb_module._embedding_service = None

        service = get_embedding_service("all-MiniLM-L6-v2")

        assert service is not None
        assert service.model_name == "all-MiniLM-L6-v2"
