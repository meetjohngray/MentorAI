# MentorAI Ingestion Guide

This guide explains how to ingest data from all supported sources into the MentorAI vector store.

## Prerequisites

- Python environment activated (`source venv/bin/activate`)
- Dependencies installed (`pip install -r requirements.txt`)
- `.env` file configured with required settings

---

## DayOne Journal

### Step 1: Export Your DayOne Journal

1. Open DayOne app
2. Go to File > Export > JSON
3. Save the export file to `backend/data/raw/dayone/`

### Step 2: Run the Ingestion Script

From the `backend` directory:

```bash
# Option 1: Auto-detect JSON file in data/raw/dayone/
python scripts/ingest_dayone.py

# Option 2: Specify path to JSON file
python scripts/ingest_dayone.py /path/to/your/journal_export.json
```

### What Gets Stored

Each chunk includes:

**Metadata**:
- `source_type`: "dayone"
- `entry_id`: Original DayOne UUID
- `entry_index`: Position in the journal
- `chunk_index` / `total_chunks`: Chunking info
- `date`: Creation date of the entry
- `tags`: Comma-separated tags from DayOne
- `has_photos` / `photo_count`: Photo info

---

## WordPress Blog

### Step 1: Export Your WordPress Site

1. In WordPress admin, go to Tools > Export
2. Choose "All content" or "Posts"
3. Download the WXR/XML export file
4. Save to `backend/data/raw/wordpress/`

### Step 2: Run the Ingestion Script

```bash
# Option 1: Auto-detect XML file in data/raw/wordpress/
python scripts/ingest_wordpress.py

# Option 2: Specify path to XML file
python scripts/ingest_wordpress.py /path/to/your/export.xml
```

### What Gets Stored

**Metadata**:
- `source_type`: "wordpress"
- `post_id`: WordPress post ID
- `title`: Post title
- `date`: Publication date
- `categories` / `tags`: Comma-separated
- `chunk_index` / `total_chunks`: Chunking info

---

## Contemplative Wisdom Texts

Wisdom texts are organized by tradition in subdirectories of `data/raw/wisdom/`.

### Directory Structure

```
data/raw/wisdom/
├── sources.json           # Metadata manifest
├── advaita/
│   ├── who_am_i.txt
│   └── self_enquiry.txt
├── buddhist/
│   ├── dhammapada.txt
│   ├── satipatthana_sutta.txt
│   ├── heart_sutra.txt
│   └── metta_sutta.txt
└── zen/
    ├── gateless_gate.txt
    ├── faith_in_mind.txt
    └── grass_roof_hermitage.txt
```

### Step 1: Get the Texts

**Option A: Automatic download**

```bash
python scripts/download_wisdom_texts.py
```

This downloads texts from public domain sources (accesstoinsight.org, sacred-texts.com, terebess.hu). Some texts (PDFs) require manual download — the script will print instructions.

Use `--force` to re-download existing files.

**Option B: Manual placement**

Place any `.txt` files in the appropriate tradition subdirectory. The ingestion script works with whatever `.txt` files are present — it doesn't require the download script.

### Step 2: Run the Ingestion Script

```bash
# Option 1: Auto-detect from data/raw/wisdom/
python scripts/ingest_wisdom.py

# Option 2: Specify path to wisdom directory
python scripts/ingest_wisdom.py /path/to/wisdom/texts/
```

### What Gets Stored

**Metadata**:
- `source_type`: "wisdom"
- `tradition`: Human-readable name (e.g., "Zen Buddhism")
- `tradition_key`: Directory name (e.g., "zen")
- `teacher`: Author/teacher name
- `text_title`: Title of the text
- `source`: Combined label (e.g., "The Gateless Gate by Wumen Huikai")
- `attribution`: Source attribution
- `chunk_index` / `total_chunks`: Chunking info

### Adding Your Own Texts

1. Create a subdirectory under `data/raw/wisdom/` for the tradition
2. Place `.txt` files in the subdirectory
3. Optionally add entries to `sources.json` for richer metadata
4. Run `python scripts/ingest_wisdom.py`

Without `sources.json` entries, the script infers metadata from directory and file names.

---

## Testing Search

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Test the search endpoint:

```bash
# Search all sources
curl "http://localhost:8000/search?q=meditation&limit=5"

# Filter by source type
curl "http://localhost:8000/search?q=mindfulness&source=wisdom"
curl "http://localhost:8000/search?q=gratitude&source=dayone"
curl "http://localhost:8000/search?q=stillness&source=wordpress"
```

Or visit the interactive API docs at http://localhost:8000/docs

## Troubleshooting

**No files found**: Make sure your export/text files are in the correct `data/raw/<source>/` directory.

**Import errors**: Ensure you're running from the `backend` directory and your virtual environment is activated.

**Out of memory**: For very large datasets, the embedding step may use significant memory. Process in smaller batches if needed.

**Reset vector store**: Delete `backend/data/chroma/` to clear all stored data, then re-run ingestion scripts.
