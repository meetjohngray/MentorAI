"""
Download wisdom texts from public domain sources.

Reads sources.json for URLs and target filenames, then downloads
and extracts plain text from each source. Idempotent by default:
skips files that already exist (use --force to re-download).

Usage:
    python scripts/download_wisdom_texts.py [--force]

Texts are saved to backend/data/raw/wisdom/<tradition>/<filename>.txt

Note: Some texts (e.g., PDFs from sriramanamaharshi.org) cannot be
automatically extracted. The script will print instructions for
manual download in those cases.
"""

import sys
import json
import re
import time
from pathlib import Path
from typing import Optional
import logging

import httpx
from bs4 import BeautifulSoup

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Be polite to servers
REQUEST_DELAY = 2  # seconds between requests
USER_AGENT = "MentorAI/1.0 (personal-use wisdom text downloader)"


def fetch_page(url: str) -> Optional[str]:
    """
    Fetch a web page and return the HTML content.

    Args:
        url: URL to fetch

    Returns:
        HTML string, or None if fetch failed
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def extract_accesstoinsight(html: str) -> str:
    """
    Extract main text content from accesstoinsight.org pages.

    Args:
        html: Raw HTML string

    Returns:
        Extracted plain text
    """
    soup = BeautifulSoup(html, "lxml")

    # The main content is usually in a div with id="F_suttas" or similar,
    # or just the main body text. Remove navigation and header elements.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Try to find the main content area
    main = soup.find("div", id="main") or soup.find("div", id="F_suttas")
    if not main:
        # Fall back to body content
        main = soup.find("body") or soup

    # Get text and clean up
    text = main.get_text(separator="\n")
    text = _clean_text(text)
    return text


def extract_sacred_texts(html: str) -> str:
    """
    Extract main text content from sacred-texts.com pages.

    Args:
        html: Raw HTML string

    Returns:
        Extracted plain text
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    # sacred-texts.com typically uses simple HTML body content
    body = soup.find("body") or soup
    text = body.get_text(separator="\n")
    text = _clean_text(text)
    return text


def extract_terebess(html: str) -> str:
    """
    Extract main text content from terebess.hu pages.

    Args:
        html: Raw HTML string

    Returns:
        Extracted plain text
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    body = soup.find("body") or soup
    text = body.get_text(separator="\n")
    text = _clean_text(text)
    return text


def _clean_text(text: str) -> str:
    """
    Clean up extracted text: normalize whitespace, remove excess blank lines.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(url: str, html: str) -> str:
    """
    Route to the appropriate extractor based on the URL domain.

    Args:
        url: The source URL
        html: Raw HTML content

    Returns:
        Extracted plain text
    """
    if "accesstoinsight.org" in url:
        return extract_accesstoinsight(html)
    elif "sacred-texts.com" in url:
        return extract_sacred_texts(html)
    elif "terebess.hu" in url:
        return extract_terebess(html)
    else:
        # Generic extraction
        soup = BeautifulSoup(html, "lxml")
        return _clean_text(soup.get_text(separator="\n"))


def download_wisdom_texts(wisdom_dir: Path, force: bool = False) -> None:
    """
    Download all texts defined in sources.json.

    Args:
        wisdom_dir: Path to the wisdom data directory
        force: If True, re-download existing files
    """
    sources_path = wisdom_dir / "sources.json"
    if not sources_path.exists():
        logger.error(f"No sources.json found at {sources_path}")
        sys.exit(1)

    with open(sources_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traditions = data.get("traditions", {})
    downloaded = 0
    skipped = 0
    failed = 0
    manual = 0

    for tradition_key, tradition_info in traditions.items():
        tradition_dir = wisdom_dir / tradition_key
        tradition_dir.mkdir(parents=True, exist_ok=True)

        display_name = tradition_info.get("display_name", tradition_key)
        logger.info(f"\n--- {display_name} ---")

        for text_info in tradition_info.get("texts", []):
            filename = text_info.get("filename", "")
            title = text_info.get("title", filename)
            url = text_info.get("url", "")
            notes = text_info.get("notes", "")

            if not filename or not url:
                logger.warning(f"Skipping entry with missing filename or URL: {text_info}")
                failed += 1
                continue

            output_path = tradition_dir / filename

            # Skip if already exists (unless --force)
            if output_path.exists() and not force:
                logger.info(f"  [SKIP] {title} - already exists")
                skipped += 1
                continue

            # Check for texts that need manual download
            if notes and "manual" in notes.lower():
                logger.info(f"  [MANUAL] {title}")
                logger.info(f"    URL: {url}")
                logger.info(f"    Note: {notes}")
                logger.info(f"    Save as: {output_path}")
                manual += 1
                continue

            # Download and extract
            logger.info(f"  Downloading: {title}...")
            html = fetch_page(url)
            if not html:
                logger.error(f"  [FAIL] Could not download {title}")
                failed += 1
                continue

            text = extract_text(url, html)
            if not text or len(text) < 100:
                logger.warning(
                    f"  [WARN] Extracted text for {title} seems too short "
                    f"({len(text)} chars). Saving anyway."
                )

            output_path.write_text(text, encoding="utf-8")
            logger.info(f"  [OK] {title} ({len(text)} chars)")
            downloaded += 1

            # Be polite
            time.sleep(REQUEST_DELAY)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary")
    logger.info(f"  Downloaded: {downloaded}")
    logger.info(f"  Skipped (existing): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Manual download needed: {manual}")
    logger.info("=" * 60)

    if manual > 0:
        logger.info(
            "\nFor texts marked [MANUAL], download them manually from the "
            "URLs shown above and save as plain .txt files in the "
            "appropriate tradition directory."
        )


def main():
    """Main entry point."""
    force = "--force" in sys.argv

    wisdom_dir = Path(__file__).parent.parent / "data" / "raw" / "wisdom"
    wisdom_dir.mkdir(parents=True, exist_ok=True)

    download_wisdom_texts(wisdom_dir, force=force)


if __name__ == "__main__":
    main()
