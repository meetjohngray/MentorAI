"""
MentorAI Backend - Main FastAPI Application

This is the entry point for the backend server.
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.services.embeddings import get_embedding_service
from app.database.vector_store import initialize_db, get_vector_store

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(
    title="MentorAI",
    description="A personal AI companion grounded in your journals and wisdom traditions",
    version="0.1.0"
)

# Allow requests from the React frontend (running on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "MentorAI backend is running"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    vector_store = get_vector_store()
    vector_store_status = "not_initialized"
    doc_count = 0

    if vector_store:
        try:
            stats = vector_store.get_collection_stats()
            doc_count = stats["total_documents"]
            vector_store_status = "ok"
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            vector_store_status = "error"

    return {
        "status": "healthy",
        "version": "0.1.0",
        "components": {
            "api": "ok",
            "database": "not_initialized",
            "vector_store": vector_store_status
        },
        "vector_store_documents": doc_count
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(5, description="Number of results to return", ge=1, le=20),
    source: str = Query(None, description="Filter by source type (dayone, wordpress)")
):
    """
    Search the personal knowledge base using semantic similarity.

    Args:
        q: Search query string
        limit: Maximum number of results to return (1-20)
        source: Optional filter by source type (dayone, wordpress)

    Returns:
        List of matching chunks with metadata and relevance scores
    """
    # Initialize vector store if needed
    vector_store = get_vector_store()
    if not vector_store:
        try:
            logger.info("Initializing vector store on first search...")
            vector_store = initialize_db(settings.chroma_path)
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise HTTPException(status_code=500, detail="Vector store not available")

    # Check if vector store has any documents
    stats = vector_store.get_collection_stats()
    if stats["total_documents"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents in vector store. Please run the ingestion script first."
        )

    # Validate source filter before any expensive operations
    where_filter = None
    if source:
        valid_sources = ["dayone", "wordpress"]
        if source.lower() not in valid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source type. Must be one of: {', '.join(valid_sources)}"
            )
        where_filter = {"source_type": source.lower()}

    try:
        # Get embedding service and embed the query
        embedding_service = get_embedding_service(settings.embedding_model)
        query_embedding = embedding_service.embed_text(q)

        # Search the vector store
        results = vector_store.search(query_embedding, n_results=limit, where=where_filter)

        # Format results
        formatted_results = []
        for idx in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][idx],
                "text": results["documents"][idx],
                "metadata": results["metadatas"][idx],
                "distance": results["distances"][idx],
                "relevance_score": 1 - results["distances"][idx]  # Convert distance to similarity
            })

        return {
            "query": q,
            "num_results": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")