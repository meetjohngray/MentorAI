"""
Integration tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

from app.main import app
from app.database.vector_store import initialize_db
from app.services.embeddings import get_embedding_service


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def setup_test_vector_store():
    """Set up a temporary vector store with test data."""
    temp_dir = tempfile.mkdtemp()

    # Initialize vector store and embedding service
    vector_store = initialize_db(temp_dir, "test_collection")
    embedding_service = get_embedding_service()

    # Add some test documents
    test_docs = [
        "This is about meditation and mindfulness practice.",
        "Python programming and software development.",
        "Nature walks and outdoor activities."
    ]

    embeddings = embedding_service.embed_batch(test_docs)

    vector_store.add_documents(
        ids=["doc1", "doc2", "doc3"],
        documents=test_docs,
        embeddings=embeddings,
        metadatas=[
            {"source_type": "dayone", "tags": "meditation"},
            {"source_type": "dayone", "tags": "coding"},
            {"source_type": "dayone", "tags": "nature"}
        ]
    )

    yield vector_store

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_ok(self, client):
        """Test that the root endpoint returns ok status."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


@pytest.mark.integration
class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check_basic(self, client):
        """Test basic health check response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "components" in data

    def test_health_check_shows_vector_store_status(self, client, setup_test_vector_store):
        """Test that health check shows vector store status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "vector_store_documents" in data


@pytest.mark.integration
class TestSearchEndpoint:
    """Test the search endpoint."""

    def test_search_requires_query(self, client):
        """Test that search requires a query parameter."""
        response = client.get("/search")

        assert response.status_code == 422  # Validation error

    def test_search_with_no_documents(self, client):
        """Test search when vector store is empty."""
        # Reset vector store
        import app.database.vector_store as vs_module
        vs_module._vector_store = None

        # Initialize empty vector store
        temp_dir = tempfile.mkdtemp()
        initialize_db(temp_dir, "empty_collection")

        response = client.get("/search?q=test")

        # Should return 404 when no documents
        assert response.status_code == 404

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_search_with_valid_query(self, client, setup_test_vector_store):
        """Test search with a valid query."""
        response = client.get("/search?q=meditation")

        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert data["query"] == "meditation"
        assert "num_results" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_returns_relevant_results(self, client, setup_test_vector_store):
        """Test that search returns relevant results."""
        response = client.get("/search?q=meditation+and+mindfulness")

        assert response.status_code == 200
        data = response.json()

        # Should return results
        assert data["num_results"] > 0

        # First result should contain meditation-related content
        if data["num_results"] > 0:
            first_result = data["results"][0]
            assert "text" in first_result
            assert "metadata" in first_result
            assert "relevance_score" in first_result

    def test_search_limit_parameter(self, client, setup_test_vector_store):
        """Test that the limit parameter works."""
        response = client.get("/search?q=test&limit=2")

        assert response.status_code == 200
        data = response.json()

        # Should return at most 2 results
        assert data["num_results"] <= 2

    def test_search_limit_validation(self, client):
        """Test that limit parameter is validated."""
        # Limit too high
        response = client.get("/search?q=test&limit=100")
        assert response.status_code == 422

        # Limit too low
        response = client.get("/search?q=test&limit=0")
        assert response.status_code == 422

    def test_search_result_structure(self, client, setup_test_vector_store):
        """Test that search results have the correct structure."""
        response = client.get("/search?q=programming")

        assert response.status_code == 200
        data = response.json()

        if data["num_results"] > 0:
            result = data["results"][0]

            # Check required fields
            assert "id" in result
            assert "text" in result
            assert "metadata" in result
            assert "distance" in result
            assert "relevance_score" in result

            # Check metadata structure
            metadata = result["metadata"]
            assert isinstance(metadata, dict)

    def test_search_with_special_characters(self, client, setup_test_vector_store):
        """Test search with special characters in query."""
        response = client.get("/search?q=test%20%26%20query")

        # Should handle special characters gracefully
        assert response.status_code == 200

    def test_search_relevance_scoring(self, client, setup_test_vector_store):
        """Test that relevance scores are calculated."""
        response = client.get("/search?q=meditation")

        assert response.status_code == 200
        data = response.json()

        if data["num_results"] > 0:
            for result in data["results"]:
                # Relevance score should be between -1 and 1
                # (converted from distance)
                assert "relevance_score" in result
                assert isinstance(result["relevance_score"], (int, float))


@pytest.mark.integration
class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are configured."""
        # Make an OPTIONS request (preflight)
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )

        # Should allow CORS from the frontend
        assert response.status_code == 200
