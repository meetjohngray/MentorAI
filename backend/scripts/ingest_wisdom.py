"""
Wisdom text ingestion script.

This script reads plain text files from the wisdom data directory,
chunks them, generates embeddings, and stores them in the ChromaDB vector store.

Text files should be organized by tradition in subdirectories:
    data/raw/wisdom/
    ├── sources.json         # Metadata manifest (optional but recommended)
    ├── advaita/
    │   ├── who_am_i.txt
    │   └── self_enquiry.txt
    ├── buddhist/
    │   ├── dhammapada.txt
    │   └── heart_sutra.txt
    └── zen/
        ├── gateless_gate.txt
        └── faith_in_mind.txt

Usage:
    python scripts/ingest_wisdom.py [path_to_wisdom_dir]

If no path is provided, it looks for text files in backend/data/raw/wisdom/
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingestion_utils import chunk_text, embed_and_store

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tradition directory name → display name mapping
TRADITION_DISPLAY_NAMES = {
    "advaita": "Advaita Vedanta",
    "buddhist": "Buddhism",
    "buddhism": "Buddhism",
    "zen": "Zen Buddhism",
    "tao": "Taoism",
    "taoism": "Taoism",
    "stoic": "Stoicism",
    "stoicism": "Stoicism",
    "christian": "Christian Contemplative",
    "sufi": "Sufism",
    "hindu": "Hinduism",
}

# Chunk sizes for wisdom texts: larger than journal entries since
# these are coherent teachings, not personal diary entries
WISDOM_TARGET_TOKENS = 800
WISDOM_MAX_TOKENS = 1000


def infer_tradition(dir_name: str) -> str:
    """
    Map a directory name to a human-readable tradition display name.

    Args:
        dir_name: Directory name (e.g., 'advaita', 'zen', 'buddhist')

    Returns:
        Human-readable tradition name (e.g., 'Advaita Vedanta', 'Zen Buddhism')
    """
    return TRADITION_DISPLAY_NAMES.get(dir_name.lower(), dir_name.title())


def parse_text_title(filename: str) -> str:
    """
    Convert a filename into a human-readable title.

    Args:
        filename: Filename without extension (e.g., 'who_am_i', 'heart_sutra')

    Returns:
        Title-cased name (e.g., 'Who Am I', 'Heart Sutra')
    """
    # Remove extension if present
    name = Path(filename).stem
    # Replace underscores and hyphens with spaces
    name = re.sub(r'[_-]+', ' ', name)
    # Title case
    return name.title()


def load_sources_metadata(wisdom_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load sources.json to provide rich metadata for wisdom texts.

    Returns a lookup from filename → metadata dict with keys:
    title, teacher, tradition, tradition_key, attribution.

    Args:
        wisdom_dir: Path to the wisdom data directory

    Returns:
        Dict mapping filename to metadata, or empty dict if sources.json not found
    """
    sources_path = wisdom_dir / "sources.json"
    if not sources_path.exists():
        logger.warning(f"No sources.json found at {sources_path}, using inferred metadata")
        return {}

    with open(sources_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup: Dict[str, Dict[str, Any]] = {}
    traditions = data.get("traditions", {})

    for tradition_key, tradition_info in traditions.items():
        display_name = tradition_info.get("display_name", infer_tradition(tradition_key))
        for text_info in tradition_info.get("texts", []):
            filename = text_info.get("filename", "")
            if filename:
                lookup[filename] = {
                    "title": text_info.get("title", parse_text_title(filename)),
                    "teacher": text_info.get("teacher", "Unknown"),
                    "tradition": display_name,
                    "tradition_key": tradition_key,
                    "attribution": text_info.get("attribution", ""),
                }

    logger.info(f"Loaded metadata for {len(lookup)} texts from sources.json")
    return lookup


def process_wisdom_file(
    file_path: Path,
    tradition: str,
    tradition_key: str,
    metadata_lookup: Dict[str, Dict[str, Any]],
    file_index: int,
) -> List[Dict[str, Any]]:
    """
    Process a single wisdom text file into chunks with metadata.

    Args:
        file_path: Path to the .txt file
        tradition: Human-readable tradition name (e.g., 'Zen Buddhism')
        tradition_key: Raw directory name (e.g., 'zen')
        metadata_lookup: Filename → metadata from sources.json
        file_index: Index for ID generation

    Returns:
        List of chunk dicts with 'id', 'text', and 'metadata' keys
    """
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning(f"Skipping empty file: {file_path}")
        return []

    # Look up metadata from sources.json, fall back to inferred values
    filename = file_path.name
    source_meta = metadata_lookup.get(filename, {})
    title = source_meta.get("title", parse_text_title(filename))
    teacher = source_meta.get("teacher", "Unknown")
    attribution = source_meta.get("attribution", "")

    # Build the "source" field that _format_wisdom_chunks reads
    if teacher and teacher != "Unknown":
        source_label = f"{title} by {teacher}"
    else:
        source_label = title

    # Chunk with larger sizes for wisdom texts
    chunks = chunk_text(text, target_tokens=WISDOM_TARGET_TOKENS, max_tokens=WISDOM_MAX_TOKENS)

    # Build text_id from filename for stable chunk IDs
    text_id = file_path.stem

    processed_chunks = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"wisdom_{tradition_key}_{text_id}_chunk_{chunk_index}"

        metadata = {
            "source_type": "wisdom",
            "tradition": tradition,
            "tradition_key": tradition_key,
            "teacher": teacher,
            "text_title": title,
            "source": source_label,
            "attribution": attribution,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
        }

        processed_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata,
        })

    return processed_chunks


def ingest_wisdom_directory(wisdom_dir: Path) -> None:
    """
    Walk the wisdom directory, process all .txt files, and store in the vector store.

    Expects structure:
        wisdom_dir/
        ├── sources.json       (optional)
        ├── tradition_a/
        │   ├── text1.txt
        │   └── text2.txt
        └── tradition_b/
            └── text3.txt

    Text files directly in wisdom_dir (not in subdirectories) are assigned
    the tradition "General".

    Args:
        wisdom_dir: Path to the wisdom data directory
    """
    logger.info(f"Starting wisdom ingestion from {wisdom_dir}")

    # Load metadata manifest
    metadata_lookup = load_sources_metadata(wisdom_dir)

    all_chunks: List[Dict[str, Any]] = []
    file_index = 0

    # Walk subdirectories (each is a tradition)
    for subdir in sorted(wisdom_dir.iterdir()):
        if not subdir.is_dir():
            continue

        tradition_key = subdir.name
        tradition = infer_tradition(tradition_key)

        txt_files = sorted(subdir.glob("*.txt"))
        if not txt_files:
            logger.info(f"No .txt files in {subdir}, skipping")
            continue

        logger.info(f"Processing tradition: {tradition} ({len(txt_files)} texts)")

        for txt_file in txt_files:
            chunks = process_wisdom_file(
                txt_file, tradition, tradition_key, metadata_lookup, file_index
            )
            all_chunks.extend(chunks)
            file_index += 1

            logger.info(
                f"  {txt_file.name}: {len(chunks)} chunks"
            )

    # Also process any .txt files directly in the wisdom directory
    top_level_files = sorted(wisdom_dir.glob("*.txt"))
    if top_level_files:
        logger.info(f"Processing {len(top_level_files)} top-level text files")
        for txt_file in top_level_files:
            chunks = process_wisdom_file(
                txt_file, "General", "general", metadata_lookup, file_index
            )
            all_chunks.extend(chunks)
            file_index += 1

    if not all_chunks:
        logger.warning(
            "No wisdom text files found. Place .txt files in subdirectories "
            f"of {wisdom_dir} (e.g., {wisdom_dir}/zen/gateless_gate.txt)"
        )
        return

    logger.info(f"Generated {len(all_chunks)} total chunks from {file_index} files")

    # Embed and store
    embed_and_store(all_chunks)


def main():
    """Main entry point."""
    try:
        if len(sys.argv) > 1:
            wisdom_dir = Path(sys.argv[1])
            if not wisdom_dir.exists():
                logger.error(f"Directory not found: {wisdom_dir}")
                sys.exit(1)
        else:
            wisdom_dir = Path(__file__).parent.parent / "data" / "raw" / "wisdom"
            wisdom_dir.mkdir(parents=True, exist_ok=True)

        ingest_wisdom_directory(wisdom_dir)

    except Exception as e:
        logger.error(f"Error during wisdom ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
