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

## Commonplace Book

A Commonplace Book is a collection of quotes, passages, and readings you've gathered over time in a separate Day One journal. These are other people's words that resonated enough for you to save.

### Step 1: Export Your Commonplace Book Journal

1. Open Day One
2. Select your **Commonplace Book journal** (not your personal journal)
3. File > Export > JSON
4. Place the export in `backend/data/raw/commonplace/`

### Step 2: Run the Ingestion Script

```bash
# Option 1: Auto-detect JSON file in data/raw/commonplace/
python scripts/ingest_commonplace.py

# Option 2: Specify path to JSON file
python scripts/ingest_commonplace.py /path/to/your/commonplace_export.json
```

### What Gets Stored

**Metadata**:
- `source_type`: "commonplace"
- `entry_id`: Original DayOne UUID
- `entry_index`: Position in the export
- `chunk_index` / `total_chunks`: Chunking info
- `date`: When you saved/collected the entry
- `tags`: Comma-separated tags from DayOne (you may have tagged by author, topic, etc.)
- `author`: Detected from attribution patterns (e.g., "— Author Name" at end of entry)
- `book_title`: Detected from attribution if present (e.g., "Author, \"Book Title\"")

### Attribution Detection

The ingestion script automatically tries to detect attribution at the end of entries. Supported patterns:

- `— Author Name` (em dash)
- `- Author Name` (hyphen)
- `~ Author Name` (tilde)
- `— Author Name, "Book Title"` (with quoted book title)
- `— Author Name (Book Title)` (with book title in parentheses)

If no attribution pattern is found, the entry is still ingested — it just won't have `author` or `book_title` metadata.

### Image Quote Extraction

Many commonplace book entries are screenshots of quotes (e.g., from apps like Waking Up, Daily Stoic). These images can be processed to extract the text using Claude Vision.

#### Step 1: Place Images in the Photos Directory

DayOne exports include a `photos/` folder with attached images. Make sure your export is in:
```
backend/data/raw/commonplace/
├── Journal.json       # The DayOne JSON export
└── photos/            # Folder with image attachments
    ├── image1.jpg
    ├── image2.png
    └── ...
```

#### Step 2: Process Images with Claude Vision

```bash
# Process all images (results are cached to avoid re-processing)
python scripts/process_commonplace_images.py

# Force re-process all images
python scripts/process_commonplace_images.py --force

# View summary of cached extractions
python scripts/process_commonplace_images.py --summary-only
```

This script:
- Uses Claude Vision (Haiku by default) to extract quotes from images
- Detects author and source/app name when visible
- Saves results to `data/processed/commonplace_images.json` for review
- Respects rate limits with configurable delay between API calls

#### Step 3: Ingest Text and Images Together

The regular ingestion script automatically includes image-extracted quotes:

```bash
# Ingest both JSON entries AND image quotes
python scripts/ingest_commonplace.py

# Ingest ONLY from image cache (no JSON required)
python scripts/ingest_commonplace.py --images-only
```

#### Image-Specific Metadata

Image-extracted quotes include additional metadata:
- `format`: "image" (to distinguish from text entries)
- `original_image`: The source filename
- `image_source`: App/source name if detected (e.g., "Waking Up")

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
curl "http://localhost:8000/search?q=perseverance&source=commonplace"
```

Or visit the interactive API docs at http://localhost:8000/docs

## Troubleshooting

**No files found**: Make sure your export/text files are in the correct `data/raw/<source>/` directory.

**Import errors**: Ensure you're running from the `backend` directory and your virtual environment is activated.

**Out of memory**: For very large datasets, the embedding step may use significant memory. Process in smaller batches if needed.

**Reset vector store**: Delete `backend/data/chroma/` to clear all stored data, then re-run ingestion scripts.
