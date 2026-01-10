"""
DayOne journal ingestion script.

This script parses a DayOne JSON export, chunks the entries,
generates embeddings, and stores them in the ChromaDB vector store.

Usage:
    python scripts/ingest_dayone.py [path_to_journal.json]

If no path is provided, it looks for JSON files in backend/data/raw/dayone/
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.embeddings import get_embedding_service
from app.database.vector_store import initialize_db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


def chunk_text(text: str, target_tokens: int = 650, max_tokens: int = 800) -> List[str]:
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
    current_chunk = []
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


def parse_dayone_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single DayOne entry.

    Args:
        entry: DayOne entry dictionary

    Returns:
        Parsed entry with standardized fields
    """
    return {
        "uuid": entry.get("uuid", ""),
        "creation_date": entry.get("creationDate", ""),
        "text": entry.get("text", ""),
        "tags": entry.get("tags", []),
        "photos": [photo.get("identifier", "") for photo in entry.get("photos", [])]
    }


def process_entry(entry_data: Dict[str, Any], entry_index: int) -> List[Dict[str, Any]]:
    """
    Process a DayOne entry into chunks with metadata.

    Args:
        entry_data: Parsed entry data
        entry_index: Index of the entry in the journal

    Returns:
        List of chunks with metadata
    """
    text = entry_data["text"]
    if not text or not text.strip():
        return []

    chunks = chunk_text(text)
    processed_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"{entry_data['uuid']}_chunk_{chunk_index}"

        metadata = {
            "source_type": "dayone",
            "entry_id": entry_data["uuid"],
            "entry_index": entry_index,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "date": entry_data["creation_date"],
            "tags": ",".join(entry_data["tags"]) if entry_data["tags"] else "",
            "has_photos": len(entry_data["photos"]) > 0,
            "photo_count": len(entry_data["photos"])
        }

        processed_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata
        })

    return processed_chunks


def ingest_dayone_export(json_path: Path) -> None:
    """
    Main ingestion function.

    Args:
        json_path: Path to DayOne JSON export file
    """
    logger.info(f"Starting DayOne ingestion from {json_path}")

    # Load JSON export
    logger.info("Loading JSON file...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data.get("entries", [])
    logger.info(f"Found {len(entries)} entries in export")

    if not entries:
        logger.warning("No entries found in export")
        return

    # Process all entries into chunks
    logger.info("Processing and chunking entries...")
    all_chunks = []
    for idx, entry in enumerate(entries):
        parsed_entry = parse_dayone_entry(entry)
        chunks = process_entry(parsed_entry, idx)
        all_chunks.extend(chunks)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(entries)} entries...")

    logger.info(f"Generated {len(all_chunks)} total chunks from {len(entries)} entries")

    if not all_chunks:
        logger.warning("No chunks generated - all entries may be empty")
        return

    # Initialize services
    logger.info("Initializing embedding service...")
    embedding_service = get_embedding_service(settings.embedding_model)

    logger.info("Initializing vector store...")
    vector_store = initialize_db(settings.chroma_path)

    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_service.embed_batch(texts, batch_size=32, show_progress=True)

    # Add to vector store
    logger.info("Adding documents to vector store...")
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

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


def find_dayone_export() -> Path:
    """
    Find a DayOne JSON export in the default location.

    Returns:
        Path to the export file

    Raises:
        FileNotFoundError: If no export found
    """
    raw_dir = Path(__file__).parent.parent / "data" / "raw" / "dayone"
    raw_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(raw_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {raw_dir}\n"
            "Please export your DayOne journal and place the JSON file there."
        )

    if len(json_files) > 1:
        logger.warning(f"Multiple JSON files found. Using: {json_files[0]}")

    return json_files[0]


def main():
    """Main entry point."""
    try:
        # Get file path from command line or find default
        if len(sys.argv) > 1:
            json_path = Path(sys.argv[1])
            if not json_path.exists():
                logger.error(f"File not found: {json_path}")
                sys.exit(1)
        else:
            json_path = find_dayone_export()

        ingest_dayone_export(json_path)

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
