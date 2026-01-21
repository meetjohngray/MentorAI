"""
Integration tests for the FastAPI endpoints.
"""

import pytest
import httpx
from pathlib import Path
import tempfile
import shutil

from app.main import app
from app.database.vector_store import initialize_db
from app.services.embeddings import get_embedding_service


@pytest.fixture
async def client():
    """Create an async test client using the modern transport style."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def setup_test_vector_store():
    """Set up a temporary vector store with test data from multiple sources."""
    temp_dir = tempfile.mkdtemp()

    # Initialize vector store and embedding service
    vector_store = initialize_db(temp_dir, "test_collection")
    embedding_service = get_embedding_service()

    # Add test documents from different sources
    test_docs = [
        "This is about meditation and mindfulness practice.",
        "Python programming and software development.",
        "Nature walks and outdoor activities.",
        "A blog post about meditation techniques.",
        "WordPress article on coding best practices."
    ]

    embeddings = embedding_service.embed_batch(test_docs)

    vector_store.add_documents(
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        documents=test_docs,
        embeddings=embeddings,
        metadatas=[
            {"source_type": "dayone", "tags": "meditation"},
            {"source_type": "dayone", "tags": "coding"},
            {"source_type": "dayone", "tags": "nature"},
            {"source_type": "wordpress", "tags": "meditation", "title": "Meditation Guide"},
            {"source_type": "wordpress", "tags": "coding", "title": "Coding Tips"}
        ]
    )

    yield vector_store

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
class TestRootEndpoint:
    """Test the root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_ok(self, client):
        """Test that the root endpoint returns ok status."""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


@pytest.mark.integration
class TestHealthEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_basic(self, client):
        """Test basic health check response."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "components" in data

    @pytest.mark.asyncio
    async def test_health_check_shows_vector_store_status(self, client, setup_test_vector_store):
        """Test that health check shows vector store status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "vector_store_documents" in data


@pytest.mark.integration
class TestSearchEndpoint:
    """Test the search endpoint."""

    @pytest.mark.asyncio
    async def test_search_requires_query(self, client):
        """Test that search requires a query parameter."""
        response = await client.get("/search")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_search_with_no_documents(self, client):
        """Test search when vector store is empty."""
        # Reset vector store
        import app.database.vector_store as vs_module
        vs_module._vector_store = None

        # Initialize empty vector store
        temp_dir = tempfile.mkdtemp()
        initialize_db(temp_dir, "empty_collection")

        response = await client.get("/search?q=test")

        # Should return 404 when no documents
        assert response.status_code == 404

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_search_with_valid_query(self, client, setup_test_vector_store):
        """Test search with a valid query."""
        response = await client.get("/search?q=meditation")

        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert data["query"] == "meditation"
        assert "num_results" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_search_returns_relevant_results(self, client, setup_test_vector_store):
        """Test that search returns relevant results."""
        response = await client.get("/search?q=meditation+and+mindfulness")

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

    @pytest.mark.asyncio
    async def test_search_limit_parameter(self, client, setup_test_vector_store):
        """Test that the limit parameter works."""
        response = await client.get("/search?q=test&limit=2")

        assert response.status_code == 200
        data = response.json()

        # Should return at most 2 results
        assert data["num_results"] <= 2

    @pytest.mark.asyncio
    async def test_search_limit_validation(self, client):
        """Test that limit parameter is validated."""
        # Limit too high
        response = await client.get("/search?q=test&limit=100")
        assert response.status_code == 422

        # Limit too low
        response = await client.get("/search?q=test&limit=0")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_result_structure(self, client, setup_test_vector_store):
        """Test that search results have the correct structure."""
        response = await client.get("/search?q=programming")

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

    @pytest.mark.asyncio
    async def test_search_with_special_characters(self, client, setup_test_vector_store):
        """Test search with special characters in query."""
        response = await client.get("/search?q=test%20%26%20query")

        # Should handle special characters gracefully
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_relevance_scoring(self, client, setup_test_vector_store):
        """Test that relevance scores are calculated."""
        response = await client.get("/search?q=meditation")

        assert response.status_code == 200
        data = response.json()

        if data["num_results"] > 0:
            for result in data["results"]:
                # Relevance score should be between -1 and 1
                # (converted from distance)
                assert "relevance_score" in result
                assert isinstance(result["relevance_score"], (int, float))

    @pytest.mark.asyncio
    async def test_search_filter_by_dayone_source(self, client, setup_test_vector_store):
        """Test filtering search results by DayOne source."""
        response = await client.get("/search?q=meditation&source=dayone")

        assert response.status_code == 200
        data = response.json()

        # All results should be from dayone
        for result in data["results"]:
            assert result["metadata"]["source_type"] == "dayone"

    @pytest.mark.asyncio
    async def test_search_filter_by_wordpress_source(self, client, setup_test_vector_store):
        """Test filtering search results by WordPress source."""
        response = await client.get("/search?q=meditation&source=wordpress")

        assert response.status_code == 200
        data = response.json()

        # All results should be from wordpress
        for result in data["results"]:
            assert result["metadata"]["source_type"] == "wordpress"

    @pytest.mark.asyncio
    async def test_search_without_source_returns_all(self, client, setup_test_vector_store):
        """Test that search without source filter returns results from all sources."""
        response = await client.get("/search?q=meditation&limit=10")

        assert response.status_code == 200
        data = response.json()

        # Should have results from both sources
        source_types = {result["metadata"]["source_type"] for result in data["results"]}
        assert "dayone" in source_types or "wordpress" in source_types

    @pytest.mark.asyncio
    async def test_search_invalid_source_returns_error(self, client, setup_test_vector_store):
        """Test that invalid source type returns 400 error."""
        response = await client.get("/search?q=meditation&source=invalid")

        assert response.status_code == 400
        data = response.json()
        assert "Invalid source type" in data["detail"]

    @pytest.mark.asyncio
    async def test_search_source_case_insensitive(self, client, setup_test_vector_store):
        """Test that source filter is case insensitive."""
        response = await client.get("/search?q=meditation&source=DayOne")

        assert response.status_code == 200
        data = response.json()

        # Should work with different casing
        for result in data["results"]:
            assert result["metadata"]["source_type"] == "dayone"


@pytest.mark.integration
class TestCORS:
    """Test CORS configuration."""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client):
        """Test that CORS headers are configured."""
        # Make an OPTIONS request (preflight)
        response = await client.options(
            "/",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )

        # Should allow CORS from the frontend
        assert response.status_code == 200
