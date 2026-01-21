"""
WordPress ingestion script.

This script parses a WordPress WXR/XML export, extracts published posts,
chunks the content, generates embeddings, and stores them in the ChromaDB vector store.

Usage:
    python scripts/ingest_wordpress.py [path_to_export.xml]

If no path is provided, it looks for XML files in backend/data/raw/wordpress/
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from bs4 import BeautifulSoup
from lxml import etree

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

# WordPress WXR namespaces
NAMESPACES = {
    'wp': 'http://wordpress.org/export/1.2/',
    'content': 'http://purl.org/rss/1.0/modules/content/',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'excerpt': 'http://wordpress.org/export/1.2/excerpt/',
}


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


def strip_html(html_content: str) -> str:
    """
    Convert HTML content to clean plain text while preserving paragraph structure.

    Args:
        html_content: HTML string to clean

    Returns:
        Plain text with paragraph breaks preserved
    """
    if not html_content:
        return ""

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'lxml')

    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer']):
        element.decompose()

    # Replace block elements with newlines to preserve structure
    block_elements = ['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                      'li', 'blockquote', 'pre', 'tr']
    for tag in soup.find_all(block_elements):
        tag.insert_before('\n\n')
        tag.insert_after('\n\n')

    # Get text content
    text = soup.get_text()

    # Clean up whitespace while preserving paragraph breaks
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def parse_wordpress_item(item: etree._Element) -> Optional[Dict[str, Any]]:
    """
    Parse a single WordPress post/item from WXR XML.

    Args:
        item: lxml Element representing a WordPress item

    Returns:
        Parsed post data or None if not a published post
    """
    # Get post type - only process posts (not pages, attachments, etc.)
    post_type_elem = item.find('wp:post_type', NAMESPACES)
    post_type = post_type_elem.text if post_type_elem is not None else None

    if post_type != 'post':
        return None

    # Get status - only process published posts
    status_elem = item.find('wp:status', NAMESPACES)
    status = status_elem.text if status_elem is not None else None

    if status != 'publish':
        return None

    # Extract post data
    post_id_elem = item.find('wp:post_id', NAMESPACES)
    post_id = post_id_elem.text if post_id_elem is not None else ""

    title_elem = item.find('title')
    title = title_elem.text if title_elem is not None and title_elem.text else ""

    # Post date - try wp:post_date first, fall back to pubDate
    date_elem = item.find('wp:post_date', NAMESPACES)
    if date_elem is not None and date_elem.text:
        post_date = date_elem.text
    else:
        pub_date_elem = item.find('pubDate')
        post_date = pub_date_elem.text if pub_date_elem is not None else ""

    # Content - in CDATA within content:encoded
    content_elem = item.find('content:encoded', NAMESPACES)
    raw_content = content_elem.text if content_elem is not None and content_elem.text else ""

    # Extract categories and tags
    categories = []
    tags = []
    for category in item.findall('category'):
        domain = category.get('domain', '')
        nicename = category.get('nicename', '')
        display_name = category.text or nicename

        if domain == 'category':
            categories.append(display_name)
        elif domain == 'post_tag':
            tags.append(display_name)

    return {
        "post_id": post_id,
        "title": title,
        "date": post_date,
        "raw_content": raw_content,
        "categories": categories,
        "tags": tags,
        "status": status
    }


def process_post(post_data: Dict[str, Any], post_index: int) -> List[Dict[str, Any]]:
    """
    Process a WordPress post into chunks with metadata.

    Args:
        post_data: Parsed post data
        post_index: Index of the post in processing order

    Returns:
        List of chunks with metadata
    """
    # Strip HTML from content
    clean_text = strip_html(post_data["raw_content"])

    if not clean_text or not clean_text.strip():
        return []

    # Prepend title to content for better context
    title = post_data["title"]
    if title:
        full_text = f"{title}\n\n{clean_text}"
    else:
        full_text = clean_text

    chunks = chunk_text(full_text)
    processed_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"wp_{post_data['post_id']}_chunk_{chunk_index}"

        metadata = {
            "source_type": "wordpress",
            "post_id": post_data["post_id"],
            "title": title,
            "post_index": post_index,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "date": post_data["date"],
            "categories": ",".join(post_data["categories"]) if post_data["categories"] else "",
            "tags": ",".join(post_data["tags"]) if post_data["tags"] else "",
        }

        processed_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata
        })

    return processed_chunks


def clean_xml_content(content: str) -> str:
    """
    Clean XML content by removing invalid characters.

    Args:
        content: Raw XML content

    Returns:
        Cleaned XML content
    """
    # Remove control characters except tab, newline, carriage return
    # XML 1.0 only allows: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
    def is_valid_xml_char(c: str) -> bool:
        codepoint = ord(c)
        return (
            codepoint == 0x9 or
            codepoint == 0xA or
            codepoint == 0xD or
            (0x20 <= codepoint <= 0xD7FF) or
            (0xE000 <= codepoint <= 0xFFFD) or
            (0x10000 <= codepoint <= 0x10FFFF)
        )

    return ''.join(c for c in content if is_valid_xml_char(c))


def parse_wxr_file(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a WordPress WXR XML export file.

    Uses lxml with recovery mode to handle malformed XML that WordPress
    sometimes exports (invalid characters, encoding issues, etc.).

    Args:
        xml_path: Path to the WXR XML file

    Returns:
        List of parsed post data dictionaries
    """
    logger.info(f"Parsing WXR file: {xml_path}")

    # Read and clean the XML content
    logger.info("Reading and cleaning XML content...")
    with open(xml_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    cleaned_content = clean_xml_content(content)

    # Parse with lxml using recovery mode for malformed XML
    parser = etree.XMLParser(recover=True, encoding='utf-8')
    try:
        root = etree.fromstring(cleaned_content.encode('utf-8'), parser)
    except etree.XMLSyntaxError as e:
        logger.error(f"XML parsing failed even with recovery mode: {e}")
        raise

    # Find all items in the channel
    channel = root.find('channel')
    if channel is None:
        logger.warning("No channel found in WXR file")
        return []

    items = channel.findall('item')
    logger.info(f"Found {len(items)} total items in export")

    posts = []
    for item in items:
        try:
            post_data = parse_wordpress_item(item)
            if post_data:
                posts.append(post_data)
        except Exception as e:
            # Log but continue on individual item parse errors
            post_id = item.find('wp:post_id', NAMESPACES)
            post_id_text = post_id.text if post_id is not None else "unknown"
            logger.warning(f"Skipping post {post_id_text} due to parse error: {e}")

    logger.info(f"Extracted {len(posts)} published posts")
    return posts


def ingest_wordpress_export(xml_path: Path) -> None:
    """
    Main ingestion function.

    Args:
        xml_path: Path to WordPress WXR XML export file
    """
    logger.info(f"Starting WordPress ingestion from {xml_path}")

    # Parse WXR file
    posts = parse_wxr_file(xml_path)

    if not posts:
        logger.warning("No published posts found in export")
        return

    # Process all posts into chunks
    logger.info("Processing and chunking posts...")
    all_chunks = []
    for idx, post in enumerate(posts):
        chunks = process_post(post, idx)
        all_chunks.extend(chunks)

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(posts)} posts...")

    logger.info(f"Generated {len(all_chunks)} total chunks from {len(posts)} posts")

    if not all_chunks:
        logger.warning("No chunks generated - all posts may be empty")
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


def find_wordpress_export() -> Path:
    """
    Find a WordPress WXR export in the default location.

    Returns:
        Path to the export file

    Raises:
        FileNotFoundError: If no export found
    """
    raw_dir = Path(__file__).parent.parent / "data" / "raw" / "wordpress"
    raw_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(raw_dir.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {raw_dir}\n"
            "Please export your WordPress site and place the WXR XML file there."
        )

    if len(xml_files) > 1:
        logger.warning(f"Multiple XML files found. Using: {xml_files[0]}")

    return xml_files[0]


def main():
    """Main entry point."""
    try:
        # Get file path from command line or find default
        if len(sys.argv) > 1:
            xml_path = Path(sys.argv[1])
            if not xml_path.exists():
                logger.error(f"File not found: {xml_path}")
                sys.exit(1)
        else:
            xml_path = find_wordpress_export()

        ingest_wordpress_export(xml_path)

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
