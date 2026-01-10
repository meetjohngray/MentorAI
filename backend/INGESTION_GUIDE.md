# DayOne Ingestion Guide

This guide explains how to ingest your DayOne journal entries into the MentorAI vector store.

## Overview

The ingestion pipeline:
1. Parses DayOne JSON export files
2. Chunks longer entries intelligently (500-800 tokens per chunk)
3. Generates embeddings using sentence-transformers
4. Stores chunks in ChromaDB for semantic search

## Prerequisites

- Python environment activated (`source venv/bin/activate`)
- Dependencies installed (`pip install -r requirements.txt`)
- `.env` file configured with required settings

## Step 1: Export Your DayOne Journal

1. Open DayOne app
2. Go to File > Export > JSON
3. Save the export file to `backend/data/raw/dayone/`

## Step 2: Run the Ingestion Script

From the `backend` directory:

```bash
# Option 1: Auto-detect JSON file in data/raw/dayone/
python scripts/ingest_dayone.py

# Option 2: Specify path to JSON file
python scripts/ingest_dayone.py /path/to/your/journal_export.json
```

The script will:
- Parse all journal entries
- Split them into semantically meaningful chunks
- Generate embeddings (this may take a few minutes for large journals)
- Store everything in ChromaDB at `backend/data/chroma/`

## Step 3: Test the Search Endpoint

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Then test the search endpoint:

```bash
# Search for entries about meditation
curl "http://localhost:8000/search?q=meditation&limit=5"

# Search for entries about gratitude
curl "http://localhost:8000/search?q=gratitude&limit=3"
```

Or visit the interactive API docs at http://localhost:8000/docs

## What Gets Stored

Each chunk includes:

**Text**: The actual journal entry text (chunked if long)

**Metadata**:
- `source_type`: Always "dayone"
- `entry_id`: Original DayOne UUID
- `entry_index`: Position in the journal
- `chunk_index`: Which chunk of the entry (0 if not split)
- `total_chunks`: How many chunks the entry was split into
- `date`: Creation date of the entry
- `tags`: Comma-separated tags from DayOne
- `has_photos`: Boolean indicating if entry has photos
- `photo_count`: Number of photos attached

## Sample Test Data

A sample export is provided at `backend/data/raw/dayone/sample_export.json` for testing purposes. It contains 6 sample entries covering meditation, self-reflection, stoicism, gratitude, boundaries, and nature.

## Troubleshooting

**No JSON files found**: Make sure your DayOne export is in `backend/data/raw/dayone/`

**Import errors**: Ensure you're running from the `backend` directory and your virtual environment is activated

**Out of memory**: For very large journals (10,000+ entries), you may need to increase your system's available memory or process entries in batches

## Next Steps

Once your journal is ingested, you can:
- Test different search queries to see how semantic search works
- Integrate the search functionality into the frontend
- Add more data sources (Kindle highlights, voice memos, etc.)
