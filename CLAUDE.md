# MentorAI - Claude Code Context

## Project Overview
MentorAI is a personal AI companion/mentor that draws on the user's own journals, blog posts, and curated wisdom and contemplative traditions to provide coaching and reflection. It acts as a mirror and accountability partner—compassionate but bluntly honest.

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
│   │   └── retrieval.py     # RAG retrieval with balanced multi-source search
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response schemas
│   ├── prompts/
│   │   └── system_prompt.py # Mentor persona system prompt
│   └── database/
│       └── vector_store.py  # ChromaDB operations
├── data/                     # All gitignored—never commit
│   ├── raw/                 # User's exports (dayone/, wordpress/, wisdom/)
│   ├── processed/           # Chunked/prepared data
│   └── chroma/              # Vector database files
├── scripts/
│   ├── ingestion_utils.py   # Shared: chunk_text, estimate_tokens, embed_and_store
│   ├── ingest_dayone.py     # DayOne journal ingestion
│   └── ingest_wordpress.py  # WordPress WXR/XML ingestion
└── tests/
    ├── conftest.py          # Shared pytest fixtures
    ├── test_api.py          # Endpoint integration tests
    ├── test_chat.py         # Chat endpoint + retrieval tests
    ├── test_config.py       # Settings, model enum tests
    ├── test_dayone_parser.py
    ├── test_embeddings.py
    ├── test_ingestion_utils.py
    ├── test_llm_service.py  # LLM service + streaming tests
    ├── test_retrieval_service.py
    ├── test_vector_store.py
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

## Coding Standards

### DRY Principles
- **Shared logic belongs in shared modules.** Ingestion scripts use `scripts/ingestion_utils.py` for chunking, embedding, and storage—never duplicate these functions.
- **All configuration lives in `app/config.py`.** Magic numbers (chunk sizes, token limits, CORS origins, etc.) must be settings, not hardcoded values. Use `from app.config import settings`.
- **Use existing services.** The `/search` endpoint delegates to `RetrievalService`—don't reimplement embedding+search logic in endpoint handlers.
- **When adding a new ingestion source**, follow the existing pattern: parse → chunk (via `ingestion_utils.chunk_text`) → build metadata dicts → call `embed_and_store()`.

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

## Current Phase
Phase 1C complete: Full chat pipeline with RAG (retrieval-augmented generation), React frontend with chat UI, balanced multi-source retrieval, 156 backend tests at 94% coverage.

## Commands

### Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload        # Run dev server (http://localhost:8000)
pytest -v                             # Run tests (156 tests)
pytest --cov=app --cov-report=html    # Run tests with coverage
python scripts/ingest_dayone.py       # Ingest DayOne journal data
python scripts/ingest_wordpress.py    # Ingest WordPress export
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
- `backend/INGESTION_GUIDE.md` — Instructions for data ingestion
- `backend/pytest.ini` — pytest configuration

## Important Notes
- Always activate venv before running Python: `source venv/bin/activate`
- ChromaDB persists to `backend/data/chroma/`—delete this folder to reset the vector store
- When adding new dependencies: update requirements.txt (Python) or package.json (JS)
- CORS origins are configurable via `CORS_ORIGINS` env var (comma-separated)
- Chunk sizes are configurable via `CHUNK_TARGET_TOKENS` / `CHUNK_MAX_TOKENS` env vars

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