"""
MentorAI Backend - Main FastAPI Application

This is the entry point for the backend server.
Run with: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.services.embeddings import get_embedding_service
from app.database.vector_store import initialize_db, get_vector_store
from app.services.retrieval import get_retrieval_service
from app.database.conversation_store import get_conversation_store
from app.routers.chat import router as chat_router
from app.routers.conversations import router as conversations_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    store = get_conversation_store()
    store.init_db()
    yield


# Create the FastAPI application
app = FastAPI(
    title="MentorAI",
    description="A personal AI companion grounded in your journals and wisdom traditions",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow requests from the React frontend (configurable via CORS_ORIGINS env var)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(conversations_router)


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
    source: str = Query(None, description="Filter by source type (dayone, wordpress, wisdom, commonplace)")
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
    if source:
        valid_sources = ["dayone", "wordpress", "wisdom", "commonplace"]
        if source.lower() not in valid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source type. Must be one of: {', '.join(valid_sources)}"
            )

    try:
        # Use RetrievalService for search
        retrieval_service = get_retrieval_service()
        result = retrieval_service.retrieve(
            query=q,
            top_k=limit,
            source_filter=source.lower() if source else None
        )

        # Format results
        formatted_results = []
        for chunk in result.chunks:
            formatted_results.append({
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "distance": chunk.distance,
                "relevance_score": chunk.relevance_score
            })

        return {
            "query": q,
            "num_results": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")