"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that gets cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_dayone_data():
    """Sample DayOne export data for testing."""
    return {
        "entries": [
            {
                "uuid": "TEST-UUID-001",
                "creationDate": "2024-01-15T10:30:00Z",
                "text": "This is a test journal entry about meditation and mindfulness. "
                       "It's a short entry that should fit in one chunk.",
                "tags": ["test", "meditation"],
                "photos": []
            },
            {
                "uuid": "TEST-UUID-002",
                "creationDate": "2024-01-20T08:15:00Z",
                "text": "Another test entry about coding.\n\n"
                       "This one has multiple paragraphs.\n\n"
                       "And talks about Python programming.",
                "tags": ["coding", "python"],
                "photos": [{"identifier": "test_photo.jpg"}]
            },
            {
                "uuid": "TEST-UUID-003",
                "creationDate": "2024-02-01T19:45:00Z",
                "text": "",  # Empty entry
                "tags": [],
                "photos": []
            }
        ]
    }


@pytest.fixture
def long_text():
    """Generate a long text that should be chunked."""
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Paragraph {i}: This is a longer piece of text that will need to be chunked. "
            f"It contains multiple sentences to make it realistic. "
            f"The chunking algorithm should split this appropriately based on paragraph boundaries. "
            f"We want to ensure that semantic meaning is preserved when chunks are created."
        )
    return "\n\n".join(paragraphs)
