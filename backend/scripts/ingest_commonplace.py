"""
Commonplace Book ingestion script.

This script parses a DayOne JSON export of the user's Commonplace Book—a collection
of quotes, passages, and readings they've gathered over time. It uses the same export
format as the personal journal but stores entries with source_type "commonplace".

The key difference from the personal journal is that these are OTHER people's words
that the user has collected. The act of collecting them is meaningful—these resonated
enough to save.

This script also ingests quotes extracted from images (via process_commonplace_images.py).
The image extraction cache at data/processed/commonplace_images.json is read and
valid quotes are included in the ingestion.

Usage:
    python scripts/ingest_commonplace.py [path_to_export.json]
    python scripts/ingest_commonplace.py --images-only  # Only ingest from image cache

If no path is provided, it looks for JSON files in backend/data/raw/commonplace/
"""

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

# Attribution patterns to detect at the end of entries
# Matches: "— Author Name", "- Author Name", "~ Author Name"
ATTRIBUTION_PATTERNS = [
    re.compile(r'\n\s*[—–-]\s*(.+?)$', re.MULTILINE),
    re.compile(r'\n\s*~\s*(.+?)$', re.MULTILINE),
]

# Patterns to detect book/source titles in attribution lines
# Matches: "Author Name, Book Title" or "Author Name (Book Title)"
BOOK_TITLE_PATTERNS = [
    re.compile(r'^(.+?),\s+["\u201c](.+?)["\u201d]$'),  # Author, "Book Title"
    re.compile(r'^(.+?)\s+\((.+?)\)$'),  # Author (Book Title)
]


def extract_attribution(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Try to extract attribution (author/source) from the entry text.

    Looks for common patterns like "— Author Name" or "- Author" at the end
    of the text.

    Args:
        text: The full entry text

    Returns:
        Tuple of (clean_text, author, book_title) where author and book_title
        may be None if not detected
    """
    author = None
    book_title = None
    clean_text = text

    for pattern in ATTRIBUTION_PATTERNS:
        match = pattern.search(text)
        if match:
            attribution_line = match.group(1).strip()
            # Remove the attribution line from the text
            clean_text = text[:match.start()].strip()

            # Try to extract book title from the attribution
            for title_pattern in BOOK_TITLE_PATTERNS:
                title_match = title_pattern.match(attribution_line)
                if title_match:
                    author = title_match.group(1).strip()
                    book_title = title_match.group(2).strip()
                    break

            if not author:
                author = attribution_line

            break

    return clean_text, author, book_title


def parse_commonplace_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single DayOne entry from the Commonplace Book.

    Args:
        entry: DayOne entry dictionary

    Returns:
        Parsed entry with standardized fields
    """
    text = entry.get("text", "")
    clean_text, author, book_title = extract_attribution(text)

    return {
        "uuid": entry.get("uuid", ""),
        "creation_date": entry.get("creationDate", ""),
        "text": text,
        "clean_text": clean_text,
        "tags": entry.get("tags", []),
        "author": author,
        "book_title": book_title,
    }


def _build_searchable_text(text: str, author: Optional[str], book_title: Optional[str]) -> str:
    """
    Build text for embedding that includes attribution for better semantic search.

    By prepending author/book info, searches like "quotes from David Whyte" will
    semantically match quotes attributed to that author.

    Args:
        text: The quote/entry text
        author: Author name if detected
        book_title: Book title if detected

    Returns:
        Text with attribution prefix for embedding
    """
    if not author:
        return text

    if book_title:
        prefix = f"[Quote by {author}, from \"{book_title}\"] "
    else:
        prefix = f"[Quote by {author}] "

    return prefix + text


def process_commonplace_entry(
    entry_data: Dict[str, Any], entry_index: int
) -> List[Dict[str, Any]]:
    """
    Process a Commonplace Book entry into chunks with metadata.

    Args:
        entry_data: Parsed entry data
        entry_index: Index of the entry in the export

    Returns:
        List of chunks with metadata
    """
    text = entry_data["text"]
    if not text or not text.strip():
        return []

    # Build searchable text with author attribution for better semantic matching
    searchable_text = _build_searchable_text(
        text, entry_data["author"], entry_data["book_title"]
    )

    chunks = chunk_text(searchable_text)
    processed_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"commonplace_{entry_data['uuid']}_chunk_{chunk_index}"

        metadata: Dict[str, Any] = {
            "source_type": "commonplace",
            "entry_id": entry_data["uuid"],
            "entry_index": entry_index,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "date": entry_data["creation_date"],
            "tags": ",".join(entry_data["tags"]) if entry_data["tags"] else "",
        }

        # Add attribution metadata if detected
        if entry_data["author"]:
            metadata["author"] = entry_data["author"]
        if entry_data["book_title"]:
            metadata["book_title"] = entry_data["book_title"]

        processed_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata,
        })

    return processed_chunks


def load_image_cache() -> Dict[str, Any]:
    """
    Load the image extraction cache.

    Returns:
        Dictionary mapping filenames to extraction results
    """
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "commonplace_images.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load image cache: {e}")
    return {}


def process_image_entry(
    filename: str, image_data: Dict[str, Any], entry_index: int
) -> List[Dict[str, Any]]:
    """
    Process an image-extracted quote into chunks with metadata.

    Args:
        filename: Original image filename
        image_data: Extraction result from cache
        entry_index: Index for this entry

    Returns:
        List of chunks with metadata
    """
    quote = image_data.get("quote")
    if not quote or not quote.strip():
        return []

    # Generate a stable ID from the filename
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:12]

    # Build searchable text with author attribution for better semantic matching
    author = image_data.get("author")
    searchable_text = _build_searchable_text(quote, author, None)

    chunks = chunk_text(searchable_text)
    processed_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"commonplace_img_{file_hash}_chunk_{chunk_index}"

        metadata: Dict[str, Any] = {
            "source_type": "commonplace",
            "entry_id": f"img_{file_hash}",
            "entry_index": entry_index,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "format": "image",
            "original_image": filename,
        }

        # Add extraction date if available
        if image_data.get("extracted_at"):
            metadata["date"] = image_data["extracted_at"]

        # Add author if detected
        if author:
            metadata["author"] = author

        # Add source (e.g., "Waking Up") as a tag or separate field
        if image_data.get("source"):
            metadata["image_source"] = image_data["source"]

        processed_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata,
        })

    return processed_chunks


def ingest_from_image_cache() -> List[Dict[str, Any]]:
    """
    Load and process quotes from the image extraction cache.

    Returns:
        List of processed chunks from image-extracted quotes
    """
    cache = load_image_cache()
    if not cache:
        logger.info("No image cache found or cache is empty")
        return []

    logger.info(f"Found {len(cache)} entries in image cache")

    all_chunks = []
    valid_count = 0
    entry_index = 0

    for filename, image_data in cache.items():
        # Skip entries with errors or no quote
        if image_data.get("error") or not image_data.get("quote"):
            continue

        valid_count += 1
        chunks = process_image_entry(filename, image_data, entry_index)
        all_chunks.extend(chunks)
        entry_index += 1

    logger.info(f"Processed {valid_count} valid image quotes into {len(all_chunks)} chunks")
    return all_chunks


def ingest_commonplace_export(json_path: Optional[Path] = None, images_only: bool = False) -> None:
    """
    Main ingestion function for the Commonplace Book.

    Args:
        json_path: Path to DayOne JSON export file (optional if images_only)
        images_only: If True, only ingest from image cache
    """
    all_chunks = []

    # Process DayOne JSON export (unless images_only)
    if not images_only and json_path:
        logger.info(f"Starting Commonplace Book ingestion from {json_path}")

        # Load JSON export
        logger.info("Loading JSON file...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = data.get("entries", [])
        logger.info(f"Found {len(entries)} entries in export")

        if entries:
            # Process all entries into chunks
            logger.info("Processing and chunking entries...")
            attribution_count = 0
            for idx, entry in enumerate(entries):
                parsed_entry = parse_commonplace_entry(entry)
                if parsed_entry["author"]:
                    attribution_count += 1

                chunks = process_commonplace_entry(parsed_entry, idx)
                all_chunks.extend(chunks)

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(entries)} entries...")

            logger.info(f"Generated {len(all_chunks)} chunks from {len(entries)} text entries")
            logger.info(f"Detected attribution in {attribution_count}/{len(entries)} entries")

    # Process image-extracted quotes
    logger.info("Checking for image-extracted quotes...")
    image_chunks = ingest_from_image_cache()
    if image_chunks:
        all_chunks.extend(image_chunks)
        logger.info(f"Added {len(image_chunks)} chunks from image extractions")

    if not all_chunks:
        logger.warning("No chunks generated - no text entries or image quotes found")
        return

    logger.info(f"Total chunks to ingest: {len(all_chunks)}")

    # Embed and store
    embed_and_store(all_chunks)


def main():
    """Main entry point."""
    try:
        # Check for --images-only flag
        images_only = "--images-only" in sys.argv
        args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

        if images_only:
            logger.info("Running in images-only mode")
            ingest_commonplace_export(json_path=None, images_only=True)
        elif args:
            # Get file path from command line
            json_path = Path(args[0])
            if not json_path.exists():
                logger.error(f"File not found: {json_path}")
                sys.exit(1)
            ingest_commonplace_export(json_path)
        else:
            # Find default file
            json_path = find_export_file("commonplace", "*.json")
            ingest_commonplace_export(json_path)

    except FileNotFoundError as e:
        # No JSON file found - check if we have images to process
        logger.warning(str(e))
        logger.info("Attempting to ingest from image cache only...")
        ingest_commonplace_export(json_path=None, images_only=True)

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
