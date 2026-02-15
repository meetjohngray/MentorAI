# MentorAI - Claude Code Context

## Project Overview
MentorAI is a personal AI companion/mentor that draws on the user's own journals, blog posts, and curated wisdom from contemplative traditions to provide coaching and reflection. It acts as a mirror and accountability partner—compassionate but bluntly honest.

## Architecture
- **Backend**: Python/FastAPI (port 8000)
- **Frontend**: React/Vite/TypeScript (port 5173)
- **Vector Database**: ChromaDB (local, file-based)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, runs locally)
- **LLM**: Claude API (user data stays local, only relevant chunks sent per query)

## Key Directories
```
backend/
├── app/                      # FastAPI application
│   ├── main.py              # Entry point, /health, /search endpoints
│   ├── config.py            # All settings from .env (single source of truth)
│   ├── routers/
│   │   └── chat.py          # POST /chat endpoint with RAG integration
│   ├── services/
│   │   ├── embeddings.py    # Text embedding with sentence-transformers
│   │   ├── llm.py           # Claude API service (sync + streaming)
│   │   └── retrieval.py     # RAG retrieval with 4-way balanced multi-source search
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response schemas
│   ├── prompts/
│   │   └── system_prompt.py # Mentor persona system prompt
│   └── database/
│       └── vector_store.py  # ChromaDB operations
├── data/                     # All gitignored—never commit
│   ├── raw/                 # User's exports (dayone/, wordpress/, wisdom/, commonplace/)
│   ├── processed/           # Chunked/prepared data
│   └── chroma/              # Vector database files
├── scripts/
│   ├── ingestion_utils.py   # Shared: chunk_text, estimate_tokens, embed_and_store
│   ├── ingest_dayone.py     # DayOne journal ingestion
│   ├── ingest_wordpress.py  # WordPress WXR/XML ingestion
│   ├── ingest_wisdom.py     # Contemplative wisdom text ingestion
│   ├── ingest_commonplace.py # Commonplace Book (collected quotes) ingestion
│   ├── process_commonplace_images.py  # Extract quotes from images via Claude Vision
│   └── download_wisdom_texts.py  # Downloads public domain wisdom texts
└── tests/
    ├── conftest.py          # Shared pytest fixtures
    ├── test_api.py          # Endpoint integration tests
    ├── test_chat.py         # Chat endpoint + retrieval tests
    ├── test_config.py       # Settings, model enum tests
    ├── test_dayone_parser.py
    ├── test_embeddings.py
    ├── test_ingestion_utils.py
    ├── test_llm_service.py  # LLM service + streaming tests
    ├── test_retrieval_service.py  # Includes wisdom + commonplace search tests
    ├── test_vector_store.py
    ├── test_wisdom_ingestion.py   # Wisdom pipeline tests
    ├── test_commonplace_ingestion.py  # Commonplace Book pipeline tests
    ├── test_image_processor.py    # Image quote extraction tests
    └── test_wordpress_parser.py

frontend/                     # React + TypeScript
├── src/
│   ├── components/          # ChatContainer, ChatInput, ChatMessage
│   ├── pages/               # ChatPage
│   ├── services/api.ts      # Typed API client
│   ├── types/index.ts       # TypeScript interfaces
│   ├── test/                # Vitest + Testing Library tests
│   ├── App.tsx
│   └── main.tsx
└── public/
```

## Four Source Types

MentorAI retrieves context from four distinct source types, each with different character:

| Source | Type Key | Ingestion Script | Chunk Sizes | Character |
|--------|----------|------------------|-------------|-----------|
| DayOne journal | `dayone` | `ingest_dayone.py` | 650/800 tokens (default) | Personal voice |
| WordPress blog | `wordpress` | `ingest_wordpress.py` | 650/800 tokens (default) | Personal voice |
| Wisdom texts | `wisdom` | `ingest_wisdom.py` | 800/1000 tokens (larger) | Curated wisdom |
| Commonplace Book | `commonplace` | `ingest_commonplace.py` | 650/800 tokens (default) | Curated wisdom |

Sources are grouped conceptually:
- **Personal voice**: `dayone`, `wordpress` — the user's own words
- **Curated wisdom**: `wisdom`, `commonplace` — wisdom the user has gathered

### Retrieval Strategy
- **General queries** (no source keywords): 35% journal, 35% blog, 15% wisdom, 15% commonplace
- **Source-prioritized queries** (detected via keywords): 80% primary source (or 60% for wisdom/commonplace), remainder split across others
- **Explicit filter**: 100% from the specified `source_filter`

### Wisdom Text Architecture
- Texts organized by tradition in `data/raw/wisdom/<tradition>/` subdirectories
- `sources.json` manifest provides rich metadata (title, teacher, tradition, attribution)
- Without `sources.json`, metadata is inferred from directory and file names
- `download_wisdom_texts.py` fetches from public domain sites but is not required—any `.txt` files work
- Current traditions: Advaita Vedanta, Buddhism, Zen Buddhism (extensible)

## Coding Standards

### DRY Principles
- **Shared logic belongs in shared modules.** Ingestion scripts use `scripts/ingestion_utils.py` for chunking, embedding, and storage—never duplicate these functions.
- **All configuration lives in `app/config.py`.** Magic numbers (chunk sizes, token limits, CORS origins, etc.) must be settings, not hardcoded values. Use `from app.config import settings`.
- **Use existing services.** The `/search` endpoint delegates to `RetrievalService`—don't reimplement embedding+search logic in endpoint handlers.
- **When adding a new ingestion source**, follow the existing pattern: parse → chunk (via `ingestion_utils.chunk_text`) → build metadata dicts → call `embed_and_store()`. Then update `RetrievalService` to include the new source in balanced/prioritized search, add to `SourcePriority` enum, add keywords, and update valid sources in `main.py`.

### Python (Backend)
- Use type hints on all function signatures—including `Optional[]` for nullable singletons
- Docstrings for all public functions (Google style)
- Pydantic models for all API request/response schemas with appropriate validation (`min_length`, `max_length`, `ge`, `le`)
- Keep functions small and single-purpose
- Singleton services (`get_embedding_service`, `get_llm_service`, `get_retrieval_service`) with `reset_*` functions for testing
- Don't leak internal error details in HTTP responses—log the full error, return a generic message

### TypeScript/React (Frontend)
- Strict TypeScript—no `any` types unless absolutely necessary
- Define interfaces for all API responses in `src/types/`
- Functional components with hooks
- Custom hooks for shared logic
- Keep components focused—extract when >100 lines

### Testing
- Write tests alongside implementation, not after
- Backend: pytest (run with `pytest -v` from backend/)
- Frontend: Vitest + Testing Library
- Test files mirror source structure: `app/services/llm.py` → `tests/test_llm_service.py`
- Aim for tests that verify behavior, not implementation details
- Mock external services (Claude API) but test with real vector stores where practical
- Target: maintain ≥90% backend coverage

## Data Privacy
- User's personal data (journals, etc.) lives ONLY in `backend/data/`
- This directory is gitignored—never commit personal content
- Only small relevant chunks are sent to Claude API per query
- No data is stored remotely
- Don't leak error details that might expose file paths or internal state

### Commonplace Book Architecture
- Uses the same DayOne JSON export format as the personal journal
- Stored in `data/raw/commonplace/` (separate export from the user's Commonplace Book journal)
- These are OTHER people's words that the user has collected — not their own writing
- Attribution is automatically extracted from patterns like "— Author Name" at end of entries
- Grouped with wisdom sources for retrieval purposes (curated wisdom the user resonates with)
- The act of collecting these quotes is itself meaningful data about the user

### Image Quote Extraction
- Many commonplace entries are screenshots of quotes (from apps like Waking Up, Daily Stoic)
- `process_commonplace_images.py` extracts text from images using Claude Vision
- Uses Haiku model by default for cost efficiency (configurable via `IMAGE_EXTRACTION_MODEL`)
- Results cached to `data/processed/commonplace_images.json` to avoid re-processing
- `ingest_commonplace.py` automatically includes cached image extractions
- Image-extracted quotes include `format: "image"` and `original_image` in metadata

## Current Phase
Phase 2C complete: Image quote extraction for Commonplace Book. Claude Vision extracts quotes from screenshot images (Waking Up, Daily Stoic, etc.). Results cached for review before ingestion. Image-extracted quotes automatically included in commonplace ingestion.

## Commands

### Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload        # Run dev server (http://localhost:8000)
pytest -v                             # Run tests (181 tests)
pytest --cov=app --cov-report=html    # Run tests with coverage
python scripts/ingest_dayone.py       # Ingest DayOne journal data
python scripts/ingest_wordpress.py    # Ingest WordPress export
python scripts/download_wisdom_texts.py        # Download wisdom texts (optional)
python scripts/download_wisdom_texts.py --force # Re-download all wisdom texts
python scripts/ingest_wisdom.py       # Ingest wisdom texts
python scripts/ingest_commonplace.py  # Ingest Commonplace Book
python scripts/process_commonplace_images.py  # Extract quotes from images (optional)
python scripts/ingest_commonplace.py --images-only  # Ingest only image-extracted quotes
```

### Frontend
```bash
cd frontend
npm run dev      # Run dev server (http://localhost:5173)
npm test         # Run tests (25 tests)
npm run build    # Production build
npm run lint     # TypeScript + ESLint checks
```

## Key Files
- `backend/.env` — Contains ANTHROPIC_API_KEY (never commit)
- `backend/.env.example` — Documented configuration template
- `backend/app/config.py` — Single source of truth for all settings
- `backend/scripts/ingestion_utils.py` — Shared ingestion logic (chunking, embedding, storage)
- `backend/data/raw/wisdom/sources.json` — Wisdom text metadata manifest
- `backend/INGESTION_GUIDE.md` — Instructions for all four data sources
- `backend/pytest.ini` — pytest configuration

## Important Notes
- Always activate venv before running Python: `source venv/bin/activate`
- ChromaDB persists to `backend/data/chroma/`—delete this folder to reset the vector store
- When adding new dependencies: update requirements.txt (Python) or package.json (JS)
- CORS origins are configurable via `CORS_ORIGINS` env var (comma-separated)
- Default chunk sizes (650/800) are configurable via `CHUNK_TARGET_TOKENS` / `CHUNK_MAX_TOKENS` env vars
- Wisdom texts use hardcoded larger chunks (800/1000) passed directly to `chunk_text()` since coherent teachings benefit from longer context windows
- The `_prioritized_search()` method is generalized to handle any number of secondary sources—adding a 5th source type only requires adding it to the `all_sources` list

## Gotchas
- `_detect_source_priority()` picks the source with the most keyword matches. Ties go to whichever appears first in the dict iteration (blog, journal, wisdom, commonplace). If a query has equal keyword matches, consider whether the tie-breaking behavior matters.
- The download script has per-site extractors (`extract_accesstoinsight`, `extract_sacred_texts`, `extract_terebess`). If a site changes layout, the extractor may need updating. The ingestion script itself is independent of the download script.
- `sources.json` is gitignored along with everything in `data/`. It's created by the project setup, not committed.

## What NOT to Do
- Don't commit anything in `backend/data/`
- Don't hardcode API keys, file paths, or magic numbers—use `app/config.py`
- Don't duplicate logic that already exists in a shared module or service
- Don't write overly clever code—clarity over brevity
- Don't skip tests for "simple" functions
- Don't make the frontend call Claude API directly—always go through backend
- Don't use `any` type in TypeScript—define proper interfaces
- Don't leak internal error details (stack traces, file paths) in API responses
- Don't add dependencies without confirming they're actually used
- Don't hardcode source type lists—when adding a new source, update all the touchpoints: `SourcePriority` enum, keyword lists, `_balanced_search()`, `_prioritized_search()`, `valid_sources` in `main.py`, the system prompt, frontend `Source` type union, `getSourceLabel()`, CSS badge colors, and `SourceChunk` schema
