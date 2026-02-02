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
from typing import List, Dict, Any
import logging

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingestion_utils import chunk_text, embed_and_store, find_export_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    # Embed and store
    embed_and_store(all_chunks)


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
            json_path = find_export_file("dayone", "*.json")

        ingest_dayone_export(json_path)

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
