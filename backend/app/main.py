"""
MentorAI Backend - Main FastAPI Application

This is the entry point for the backend server.
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return {
        "status": "healthy",
        "version": "0.1.0",
        "components": {
            "api": "ok",
            "database": "not_initialized",  # We'll update this later
            "vector_store": "not_initialized"
        }
    }