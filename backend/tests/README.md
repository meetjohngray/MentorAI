# Test Suite for MentorAI Backend

This directory contains comprehensive unit and integration tests for the MentorAI backend.

## Test Coverage

**51 tests** covering **88% of the codebase**

### Test Files

1. **[test_embeddings.py](test_embeddings.py)** (9 tests)
   - Tests for the embedding service using sentence-transformers
   - Validates embedding generation, consistency, and similarity
   - Tests singleton pattern implementation

2. **[test_vector_store.py](test_vector_store.py)** (11 tests)
   - Tests for ChromaDB vector store operations
   - Document addition, search, filtering, and persistence
   - Collection management and statistics

3. **[test_dayone_parser.py](test_dayone_parser.py)** (18 tests)
   - Tests for DayOne JSON parsing and chunking
   - Token estimation and intelligent text chunking
   - Entry processing with metadata extraction

4. **[test_api.py](test_api.py)** (13 tests)
   - Integration tests for FastAPI endpoints
   - Health checks, search functionality, CORS
   - Request validation and error handling

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage Report
```bash
pytest --cov=app --cov-report=html
```

Then open `htmlcov/index.html` in a browser to see detailed coverage.

### Run Specific Test Files
```bash
# Unit tests only
pytest tests/test_embeddings.py tests/test_vector_store.py tests/test_dayone_parser.py

# API tests only
pytest tests/test_api.py
```

### Run Tests by Marker
```bash
# Unit tests
pytest -m unit

# Integration tests
pytest -m integration
```

### Run with Verbose Output
```bash
pytest -v
```

### Run and Stop on First Failure
```bash
pytest -x
```

## Test Configuration

Configuration is in [pytest.ini](../pytest.ini):
- Test discovery patterns
- Coverage settings
- Asyncio mode
- Custom markers

## Fixtures

Shared fixtures are defined in [conftest.py](conftest.py):
- `temp_dir`: Temporary directory for tests
- `sample_dayone_data`: Sample DayOne export data
- `long_text`: Long text for chunking tests

## Code Coverage

Current coverage by module:
- `app/services/embeddings.py`: **100%**
- `app/config.py`: **100%**
- `app/database/vector_store.py`: **93%**
- `app/main.py`: **76%**

**Overall: 88%**

## Writing New Tests

### Unit Test Example
```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_basic_functionality(self):
        # Arrange
        input_data = "test"

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected_output
```

### Integration Test Example
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.integration
class TestMyEndpoint:
    def test_endpoint_returns_success(self):
        client = TestClient(app)
        response = client.get("/my-endpoint")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
```

## Dependencies

Testing dependencies (from requirements.txt):
- `pytest==7.4.4`: Test framework
- `pytest-asyncio==0.23.3`: Async test support
- `pytest-cov==4.1.0`: Coverage reporting
- `pytest-mock==3.12.0`: Mocking utilities

## Continuous Integration

Tests should be run before:
- Committing changes
- Creating pull requests
- Deploying to production

## Troubleshooting

### Tests Fail Due to Import Errors
Ensure you're in the backend directory and the virtual environment is activated:
```bash
cd backend
source venv/bin/activate
pytest
```

### ChromaDB Persistence Issues
Some tests create temporary ChromaDB instances. If you encounter issues, try:
```bash
rm -rf .pytest_cache
pytest --cache-clear
```

### Embedding Model Download
First-time test runs will download the sentence-transformer model (~90MB). This is normal and only happens once.
