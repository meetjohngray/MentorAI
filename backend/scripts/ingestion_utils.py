"""
Shared utilities for ingestion scripts.

Contains common functions used by both DayOne and WordPress ingestion pipelines:
- Text chunking and token estimation
- Embedding generation and vector store insertion
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from app.config import settings
from app.services.embeddings import get_embedding_service
from app.database.vector_store import initialize_db

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (assuming ~4 chars per token).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_text(
    text: str,
    target_tokens: int = settings.chunk_target_tokens,
    max_tokens: int = settings.chunk_max_tokens,
) -> List[str]:
    """
    Split text into chunks, preferring paragraph boundaries.

    Args:
        text: Text to chunk
        target_tokens: Target tokens per chunk
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_tokens = estimate_tokens(paragraph)

        # If single paragraph exceeds max, split it on sentences
        if para_tokens > max_tokens:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                sent_tokens = estimate_tokens(sentence)
                if current_tokens + sent_tokens > target_tokens and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sent_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
        else:
            # Add paragraph to current chunk if it fits
            if current_tokens + para_tokens > target_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def embed_and_store(
    chunks: List[Dict[str, Any]],
    batch_size: int = 32,
) -> None:
    """
    Generate embeddings for chunks and store them in the vector store.

    Args:
        chunks: List of dicts with 'id', 'text', and 'metadata' keys
        batch_size: Number of texts to embed at once
    """
    # Initialize services
    logger.info("Initializing embedding service...")
    embedding_service = get_embedding_service(settings.embedding_model)

    logger.info("Initializing vector store...")
    vector_store = initialize_db(settings.chroma_path)

    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_batch(texts, batch_size=batch_size, show_progress=True)

    # Add to vector store
    logger.info("Adding documents to vector store...")
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vector_store.add_documents(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # Print stats
    stats = vector_store.get_collection_stats()
    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"Total documents in vector store: {stats['total_documents']}")
    logger.info(f"Persist directory: {stats['persist_directory']}")
    logger.info("=" * 60)


def find_export_file(raw_subdir: str, extension: str) -> Path:
    """
    Find an export file in the default data/raw location.

    Args:
        raw_subdir: Subdirectory name under data/raw/ (e.g., 'dayone', 'wordpress')
        extension: File extension to search for (e.g., '*.json', '*.xml')

    Returns:
        Path to the export file

    Raises:
        FileNotFoundError: If no matching files found
    """
    raw_dir = Path(__file__).parent.parent / "data" / "raw" / raw_subdir
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.glob(extension))

    if not files:
        raise FileNotFoundError(
            f"No {extension} files found in {raw_dir}\n"
            f"Please place your export file there."
        )

    if len(files) > 1:
        logger.warning(f"Multiple {extension} files found. Using: {files[0]}")

    return files[0]
