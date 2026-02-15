#!/usr/bin/env python3
"""
Process images from the Commonplace Book photos folder.

This script extracts quotes from images using Claude Vision and saves the results
to a JSON cache file. The cache prevents re-processing already-extracted images
and allows review before ingestion.

Usage:
    python scripts/process_commonplace_images.py [--force] [--photos-dir PATH]

Options:
    --force         Re-process all images, ignoring cache
    --photos-dir    Path to photos directory (default: data/raw/commonplace/photos/)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.image_processor import (
    get_image_processor,
    SUPPORTED_FORMATS,
    ImageExtractionResult,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PHOTOS_DIR = Path(__file__).parent.parent / "data" / "raw" / "commonplace" / "photos"
DEFAULT_CACHE_FILE = Path(__file__).parent.parent / "data" / "processed" / "commonplace_images.json"


def load_cache(cache_path: Path) -> Dict[str, Any]:
    """
    Load existing extraction cache.

    Args:
        cache_path: Path to the cache file

    Returns:
        Dictionary mapping filenames to extraction results
    """
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load cache: {e}")
    return {}


def save_cache(cache: Dict[str, Any], cache_path: Path) -> None:
    """
    Save extraction cache to file.

    Args:
        cache: Dictionary mapping filenames to extraction results
        cache_path: Path to the cache file
    """
    # Ensure directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    logger.info(f"Cache saved to {cache_path}")


def find_images(photos_dir: Path) -> list[Path]:
    """
    Find all supported image files in the photos directory.

    Args:
        photos_dir: Path to the photos directory

    Returns:
        List of paths to image files
    """
    if not photos_dir.exists():
        logger.warning(f"Photos directory does not exist: {photos_dir}")
        return []

    images = []
    for ext in SUPPORTED_FORMATS:
        # Case-insensitive matching
        images.extend(photos_dir.glob(f"*{ext}"))
        images.extend(photos_dir.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def process_images(
    photos_dir: Path,
    cache_path: Path,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Process all images in the photos directory.

    Args:
        photos_dir: Path to the photos directory
        cache_path: Path to the cache file
        force: If True, re-process all images

    Returns:
        Dictionary mapping filenames to extraction results
    """
    # Load existing cache
    cache = {} if force else load_cache(cache_path)

    # Find images
    images = find_images(photos_dir)
    logger.info(f"Found {len(images)} images in {photos_dir}")

    if not images:
        return cache

    # Initialize processor
    processor = get_image_processor()

    # Track statistics
    processed_count = 0
    skipped_count = 0
    success_count = 0
    error_count = 0

    # Process each image
    for idx, image_path in enumerate(images, 1):
        filename = image_path.name

        # Skip if already in cache (unless force)
        if filename in cache and not force:
            logger.debug(f"Skipping {filename} (already cached)")
            skipped_count += 1
            continue

        logger.info(f"Processing image {idx}/{len(images)}: {filename}")
        processed_count += 1

        # Extract quote from image
        result = processor.extract_quote_from_image(image_path)

        # Store result in cache
        cache[filename] = {
            "quote": result.quote,
            "author": result.author,
            "source": result.source,
            "confidence": result.confidence,
            "error": result.error,
            "extracted_at": datetime.utcnow().isoformat() + "Z",
            "file_path": str(image_path.relative_to(photos_dir.parent)),
        }

        if result.is_valid:
            success_count += 1
            author_info = f" by {result.author}" if result.author else ""
            logger.info(f"  -> Extracted quote{author_info}")
            if result.quote:
                # Log first 60 chars of quote
                preview = result.quote[:60] + "..." if len(result.quote) > 60 else result.quote
                logger.info(f"     \"{preview}\"")
        else:
            error_count += 1
            logger.warning(f"  -> Failed: {result.error or 'No quote detected'}")

        # Save cache periodically (every 10 images)
        if processed_count % 10 == 0:
            save_cache(cache, cache_path)

    # Final save
    save_cache(cache, cache_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"  Total images found: {len(images)}")
    logger.info(f"  Skipped (cached): {skipped_count}")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Successful extractions: {success_count}")
    logger.info(f"  Failed/no quote: {error_count}")
    logger.info("=" * 60)

    return cache


def print_summary(cache: Dict[str, Any]) -> None:
    """
    Print a summary of the extraction cache.

    Args:
        cache: Dictionary mapping filenames to extraction results
    """
    valid_count = sum(
        1 for v in cache.values()
        if v.get("quote") and len(v["quote"].strip()) > 0
    )
    with_author = sum(1 for v in cache.values() if v.get("author"))
    with_source = sum(1 for v in cache.values() if v.get("source"))
    errors = sum(1 for v in cache.values() if v.get("error"))

    logger.info("Cache Summary:")
    logger.info(f"  Total entries: {len(cache)}")
    logger.info(f"  Valid quotes: {valid_count}")
    logger.info(f"  With author: {with_author}")
    logger.info(f"  With source: {with_source}")
    logger.info(f"  Errors: {errors}")

    # List authors found
    authors = {v.get("author") for v in cache.values() if v.get("author")}
    if authors:
        logger.info(f"  Unique authors: {len(authors)}")
        for author in sorted(authors)[:10]:  # Show first 10
            logger.info(f"    - {author}")
        if len(authors) > 10:
            logger.info(f"    ... and {len(authors) - 10} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Commonplace Book images to extract quotes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all images, ignoring cache",
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=DEFAULT_PHOTOS_DIR,
        help=f"Path to photos directory (default: {DEFAULT_PHOTOS_DIR})",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help=f"Path to cache file (default: {DEFAULT_CACHE_FILE})",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing cache, don't process",
    )

    args = parser.parse_args()

    try:
        if args.summary_only:
            cache = load_cache(args.cache_file)
            if cache:
                print_summary(cache)
            else:
                logger.info("No cache file found or cache is empty")
        else:
            cache = process_images(
                photos_dir=args.photos_dir,
                cache_path=args.cache_file,
                force=args.force,
            )
            if cache:
                print_summary(cache)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
