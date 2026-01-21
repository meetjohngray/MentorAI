"""
Unit tests for the WordPress ingestion script.
"""

import pytest
from lxml import etree
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ingest_wordpress import (
    estimate_tokens,
    chunk_text,
    strip_html,
    parse_wordpress_item,
    process_post,
    parse_wxr_file,
    clean_xml_content,
    NAMESPACES
)


# Sample WXR XML for testing
SAMPLE_WXR = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
    xmlns:excerpt="http://wordpress.org/export/1.2/excerpt/"
    xmlns:content="http://purl.org/rss/1.0/modules/content/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:wp="http://wordpress.org/export/1.2/"
>
<channel>
    <title>My Blog</title>
    <item>
        <title>Test Post Title</title>
        <wp:post_id>123</wp:post_id>
        <wp:post_date>2023-05-15 10:30:00</wp:post_date>
        <wp:post_type>post</wp:post_type>
        <wp:status>publish</wp:status>
        <content:encoded><![CDATA[<p>This is the first paragraph.</p>
<p>This is the second paragraph with <strong>bold</strong> text.</p>
<blockquote>A meaningful quote.</blockquote>]]></content:encoded>
        <category domain="category" nicename="life">Life</category>
        <category domain="category" nicename="thoughts">Thoughts</category>
        <category domain="post_tag" nicename="reflection">reflection</category>
        <category domain="post_tag" nicename="personal">personal</category>
    </item>
    <item>
        <title>Draft Post</title>
        <wp:post_id>124</wp:post_id>
        <wp:post_date>2023-05-16 14:00:00</wp:post_date>
        <wp:post_type>post</wp:post_type>
        <wp:status>draft</wp:status>
        <content:encoded><![CDATA[<p>This is a draft.</p>]]></content:encoded>
    </item>
    <item>
        <title>A Page</title>
        <wp:post_id>125</wp:post_id>
        <wp:post_date>2023-05-17 09:00:00</wp:post_date>
        <wp:post_type>page</wp:post_type>
        <wp:status>publish</wp:status>
        <content:encoded><![CDATA[<p>This is a page, not a post.</p>]]></content:encoded>
    </item>
    <item>
        <title>Another Published Post</title>
        <wp:post_id>126</wp:post_id>
        <wp:post_date>2023-06-01 08:15:00</wp:post_date>
        <wp:post_type>post</wp:post_type>
        <wp:status>publish</wp:status>
        <content:encoded><![CDATA[<p>Second published post content.</p>]]></content:encoded>
        <category domain="category" nicename="tech">Tech</category>
    </item>
</channel>
</rss>
"""


@pytest.fixture
def sample_wxr_file(tmp_path):
    """Create a temporary WXR file for testing."""
    wxr_file = tmp_path / "export.xml"
    wxr_file.write_text(SAMPLE_WXR, encoding='utf-8')
    return wxr_file


@pytest.fixture
def sample_item():
    """Create a sample WordPress item Element."""
    root = etree.fromstring(SAMPLE_WXR.encode('utf-8'))
    channel = root.find('channel')
    return channel.findall('item')[0]  # First item (published post)


@pytest.fixture
def draft_item():
    """Create a draft WordPress item Element."""
    root = etree.fromstring(SAMPLE_WXR.encode('utf-8'))
    channel = root.find('channel')
    return channel.findall('item')[1]  # Second item (draft)


@pytest.fixture
def page_item():
    """Create a page WordPress item Element."""
    root = etree.fromstring(SAMPLE_WXR.encode('utf-8'))
    channel = root.find('channel')
    return channel.findall('item')[2]  # Third item (page)


@pytest.mark.unit
class TestCleanXmlContent:
    """Test the XML content cleaning function."""

    def test_clean_removes_control_characters(self):
        """Test that invalid control characters are removed."""
        # Contains NULL (0x00), BEL (0x07), and other control chars
        dirty = "Hello\x00World\x07Test\x1fEnd"
        cleaned = clean_xml_content(dirty)
        assert cleaned == "HelloWorldTestEnd"

    def test_clean_preserves_valid_whitespace(self):
        """Test that tabs, newlines, and carriage returns are preserved."""
        text = "Hello\tWorld\nNew\rLine"
        cleaned = clean_xml_content(text)
        assert cleaned == text

    def test_clean_preserves_normal_text(self):
        """Test that normal text is unchanged."""
        text = "This is normal text with punctuation! And numbers: 123."
        cleaned = clean_xml_content(text)
        assert cleaned == text

    def test_clean_handles_unicode(self):
        """Test that valid unicode is preserved."""
        text = "Hello 世界 émoji 🎉"
        cleaned = clean_xml_content(text)
        assert cleaned == text


@pytest.mark.unit
class TestEstimateTokens:
    """Test the token estimation function."""

    def test_estimate_tokens_empty_string(self):
        """Test estimating tokens for an empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test estimating tokens for short text."""
        text = "This is a test."
        estimated = estimate_tokens(text)
        assert estimated == len(text) // 4

    def test_estimate_tokens_long_text(self):
        """Test estimating tokens for longer text."""
        text = "a" * 1000
        estimated = estimate_tokens(text)
        assert estimated == 250  # 1000 / 4


@pytest.mark.unit
class TestChunkText:
    """Test the text chunking function."""

    def test_chunk_short_text(self):
        """Test that short text is not chunked."""
        text = "This is a short text that fits in one chunk."
        chunks = chunk_text(text, target_tokens=100, max_tokens=150)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text_by_paragraphs(self):
        """Test chunking long text by paragraphs."""
        paragraphs = [
            "Paragraph one. " * 50,
            "Paragraph two. " * 50,
            "Paragraph three. " * 50
        ]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, target_tokens=50, max_tokens=100)

        assert len(chunks) > 1
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, target_tokens=10, max_tokens=20)

        reconstructed = " ".join(chunks).replace("\n\n", " ")

        assert "Paragraph one" in reconstructed
        assert "Paragraph two" in reconstructed
        assert "Paragraph three" in reconstructed

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", target_tokens=100, max_tokens=150)
        assert len(chunks) <= 1


@pytest.mark.unit
class TestStripHtml:
    """Test the HTML stripping function."""

    def test_strip_simple_html(self):
        """Test stripping simple HTML tags."""
        html = "<p>Hello world</p>"
        result = strip_html(html)
        assert "Hello world" in result
        assert "<p>" not in result

    def test_strip_nested_html(self):
        """Test stripping nested HTML tags."""
        html = "<div><p>Text with <strong>bold</strong> and <em>italic</em>.</p></div>"
        result = strip_html(html)
        assert "Text with bold and italic." in result
        assert "<" not in result

    def test_preserve_paragraph_structure(self):
        """Test that paragraph breaks are preserved."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = strip_html(html)
        # Should have some separation between paragraphs
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_strip_script_and_style(self):
        """Test that script and style tags are removed."""
        html = """
        <p>Content</p>
        <script>alert('bad');</script>
        <style>.foo { color: red; }</style>
        <p>More content</p>
        """
        result = strip_html(html)
        assert "Content" in result
        assert "More content" in result
        assert "alert" not in result
        assert "color" not in result

    def test_strip_empty_html(self):
        """Test stripping empty HTML."""
        assert strip_html("") == ""
        assert strip_html(None) == ""

    def test_strip_html_with_links(self):
        """Test stripping HTML with links."""
        html = '<p>Check out <a href="https://example.com">this link</a> for more.</p>'
        result = strip_html(html)
        assert "Check out this link for more" in result
        assert "href" not in result

    def test_strip_html_with_lists(self):
        """Test stripping HTML with lists."""
        html = "<ul><li>Item one</li><li>Item two</li></ul>"
        result = strip_html(html)
        assert "Item one" in result
        assert "Item two" in result

    def test_strip_html_with_blockquote(self):
        """Test stripping HTML with blockquotes."""
        html = "<blockquote>A wise quote from someone.</blockquote>"
        result = strip_html(html)
        assert "A wise quote from someone" in result

    def test_strip_html_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        html = "<p>Text    with   lots    of   spaces.</p>"
        result = strip_html(html)
        # Multiple spaces should be collapsed
        assert "    " not in result


@pytest.mark.unit
class TestParseWordPressItem:
    """Test parsing WordPress items from WXR XML."""

    def test_parse_published_post(self, sample_item):
        """Test parsing a published post."""
        result = parse_wordpress_item(sample_item)

        assert result is not None
        assert result["post_id"] == "123"
        assert result["title"] == "Test Post Title"
        assert result["date"] == "2023-05-15 10:30:00"
        assert result["status"] == "publish"
        assert "first paragraph" in result["raw_content"]
        assert result["categories"] == ["Life", "Thoughts"]
        assert result["tags"] == ["reflection", "personal"]

    def test_skip_draft_post(self, draft_item):
        """Test that draft posts are skipped."""
        result = parse_wordpress_item(draft_item)
        assert result is None

    def test_skip_page(self, page_item):
        """Test that pages are skipped."""
        result = parse_wordpress_item(page_item)
        assert result is None

    def test_parse_post_without_categories(self):
        """Test parsing a post without categories or tags."""
        xml = """
        <item xmlns:wp="http://wordpress.org/export/1.2/"
              xmlns:content="http://purl.org/rss/1.0/modules/content/">
            <title>Simple Post</title>
            <wp:post_id>999</wp:post_id>
            <wp:post_date>2023-07-01 12:00:00</wp:post_date>
            <wp:post_type>post</wp:post_type>
            <wp:status>publish</wp:status>
            <content:encoded><![CDATA[<p>Simple content.</p>]]></content:encoded>
        </item>
        """
        item = etree.fromstring(xml.encode('utf-8'))
        result = parse_wordpress_item(item)

        assert result is not None
        assert result["categories"] == []
        assert result["tags"] == []


@pytest.mark.unit
class TestProcessPost:
    """Test processing WordPress posts into chunks."""

    def test_process_simple_post(self):
        """Test processing a simple post."""
        post_data = {
            "post_id": "123",
            "title": "My Test Post",
            "date": "2023-05-15 10:30:00",
            "raw_content": "<p>This is a simple test post.</p>",
            "categories": ["Life"],
            "tags": ["test"],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=0)

        assert len(chunks) == 1
        assert chunks[0]["id"] == "wp_123_chunk_0"
        assert "My Test Post" in chunks[0]["text"]
        assert "simple test post" in chunks[0]["text"]
        assert chunks[0]["metadata"]["source_type"] == "wordpress"
        assert chunks[0]["metadata"]["post_id"] == "123"
        assert chunks[0]["metadata"]["title"] == "My Test Post"
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1
        assert chunks[0]["metadata"]["categories"] == "Life"
        assert chunks[0]["metadata"]["tags"] == "test"

    def test_process_post_with_multiple_categories(self):
        """Test processing a post with multiple categories and tags."""
        post_data = {
            "post_id": "456",
            "title": "Multi-tag Post",
            "date": "2023-06-01 08:00:00",
            "raw_content": "<p>Post with multiple tags.</p>",
            "categories": ["Tech", "Life", "Work"],
            "tags": ["coding", "reflection", "growth"],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=5)

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["categories"] == "Tech,Life,Work"
        assert chunks[0]["metadata"]["tags"] == "coding,reflection,growth"
        assert chunks[0]["metadata"]["post_index"] == 5

    def test_process_empty_post(self):
        """Test processing a post with empty content."""
        post_data = {
            "post_id": "789",
            "title": "Empty Post",
            "date": "2023-06-15 12:00:00",
            "raw_content": "",
            "categories": [],
            "tags": [],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=0)
        assert len(chunks) == 0

    def test_process_post_with_whitespace_only(self):
        """Test processing a post with only whitespace content."""
        post_data = {
            "post_id": "790",
            "title": "",
            "date": "2023-06-15 12:00:00",
            "raw_content": "<p>   </p><p>\n\t</p>",
            "categories": [],
            "tags": [],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=0)
        assert len(chunks) == 0

    def test_process_long_post(self, long_text):
        """Test processing a long post that needs chunking."""
        post_data = {
            "post_id": "999",
            "title": "Long Post",
            "date": "2023-07-01 10:00:00",
            "raw_content": f"<p>{long_text}</p>",
            "categories": ["Essays"],
            "tags": ["long-form"],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=10)

        assert len(chunks) > 1

        for i, chunk in enumerate(chunks):
            assert chunk["id"] == f"wp_999_chunk_{i}"
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)
            assert chunk["metadata"]["post_index"] == 10

    def test_chunk_ids_are_unique(self, long_text):
        """Test that chunk IDs are unique."""
        post_data = {
            "post_id": "unique-test",
            "title": "Unique IDs Post",
            "date": "2023-07-15 14:00:00",
            "raw_content": f"<p>{long_text}</p>",
            "categories": [],
            "tags": [],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=0)
        chunk_ids = [chunk["id"] for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.unit
class TestParseWxrFile:
    """Test parsing complete WXR files."""

    def test_parse_wxr_file(self, sample_wxr_file):
        """Test parsing a WXR file."""
        posts = parse_wxr_file(sample_wxr_file)

        # Should only get published posts (2 out of 4 items)
        assert len(posts) == 2

        # First post
        assert posts[0]["post_id"] == "123"
        assert posts[0]["title"] == "Test Post Title"
        assert len(posts[0]["categories"]) == 2
        assert len(posts[0]["tags"]) == 2

        # Second post
        assert posts[1]["post_id"] == "126"
        assert posts[1]["title"] == "Another Published Post"

    def test_parse_empty_wxr(self, tmp_path):
        """Test parsing a WXR file with no posts."""
        empty_wxr = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0" xmlns:wp="http://wordpress.org/export/1.2/">
        <channel>
            <title>Empty Blog</title>
        </channel>
        </rss>
        """
        wxr_file = tmp_path / "empty.xml"
        wxr_file.write_text(empty_wxr, encoding='utf-8')

        posts = parse_wxr_file(wxr_file)
        assert len(posts) == 0

    def test_parse_wxr_with_only_drafts(self, tmp_path):
        """Test parsing a WXR file with only draft posts."""
        draft_wxr = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"
            xmlns:wp="http://wordpress.org/export/1.2/"
            xmlns:content="http://purl.org/rss/1.0/modules/content/">
        <channel>
            <title>Draft Blog</title>
            <item>
                <title>Draft One</title>
                <wp:post_id>1</wp:post_id>
                <wp:post_type>post</wp:post_type>
                <wp:status>draft</wp:status>
                <content:encoded><![CDATA[<p>Draft content.</p>]]></content:encoded>
            </item>
        </channel>
        </rss>
        """
        wxr_file = tmp_path / "drafts.xml"
        wxr_file.write_text(draft_wxr, encoding='utf-8')

        posts = parse_wxr_file(wxr_file)
        assert len(posts) == 0


@pytest.mark.integration
class TestWordPressIntegration:
    """Integration tests for WordPress ingestion."""

    def test_full_processing_pipeline(self, sample_wxr_file):
        """Test the full processing pipeline from XML to chunks."""
        posts = parse_wxr_file(sample_wxr_file)

        all_chunks = []
        for idx, post in enumerate(posts):
            chunks = process_post(post, idx)
            all_chunks.extend(chunks)

        # Should have chunks from 2 published posts
        assert len(all_chunks) >= 2

        # Verify all chunks have required fields
        for chunk in all_chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["source_type"] == "wordpress"
            assert "post_id" in chunk["metadata"]
            assert "date" in chunk["metadata"]

    def test_html_to_searchable_text(self):
        """Test that HTML content becomes searchable plain text."""
        html_content = """
        <h1>My Journey</h1>
        <p>Today I learned something <strong>important</strong> about life.</p>
        <ul>
            <li>First lesson</li>
            <li>Second lesson</li>
        </ul>
        <blockquote>As the wise person said...</blockquote>
        """

        post_data = {
            "post_id": "html-test",
            "title": "HTML Test",
            "date": "2023-08-01 10:00:00",
            "raw_content": html_content,
            "categories": [],
            "tags": [],
            "status": "publish"
        }

        chunks = process_post(post_data, post_index=0)

        assert len(chunks) >= 1
        text = chunks[0]["text"]

        # All meaningful content should be present
        assert "My Journey" in text
        assert "important" in text
        assert "First lesson" in text
        assert "Second lesson" in text
        assert "wise person" in text

        # No HTML should remain
        assert "<" not in text
        assert ">" not in text
