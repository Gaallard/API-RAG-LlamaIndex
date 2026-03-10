"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def test_api_key():
    """Test API key for authentication."""
    return "test-api-key-12345"


@pytest.fixture
def test_headers(test_api_key):
    """Headers with test API key."""
    return {"X-API-Key": test_api_key}


@pytest.fixture(autouse=True)
def mock_settings(test_api_key, monkeypatch):
    """Mock settings for tests."""
    monkeypatch.setattr(settings, "api_key", test_api_key)
    monkeypatch.setattr(settings, "openai_api_key", None)

    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr(settings, "data_dir", Path(temp_dir))

    yield

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Mock embedding model to avoid OpenAI API calls."""
    mock_embedding = MagicMock()
    mock_embedding.get_text_embedding.return_value = [0.1] * 1536
    mock_embedding.get_text_embedding_dimension.return_value = 1536

    def get_mock_embedding():
        return mock_embedding

    monkeypatch.setattr(
        "app.services.ingest_service.get_embedding_model",
        get_mock_embedding,
    )
    monkeypatch.setattr(
        "app.services.query_service.get_embedding_model",
        get_mock_embedding,
    )

    yield mock_embedding


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM to avoid OpenAI API calls."""
    from llama_index.core.base.llms.types import CompletionResponse

    mock_llm_obj = MagicMock()

    mock_response = CompletionResponse(
        text=(
            "Machine learning is a subset of artificial intelligence "
            "that enables systems to learn and improve from experience "
            "without being explicitly programmed."
        )
    )
    mock_llm_obj.complete.return_value = mock_response

    class MockStreamDelta:
        def __init__(self, text):
            self.delta = text

    mock_llm_obj.stream_complete.return_value = [
        MockStreamDelta("Machine "),
        MockStreamDelta("learning "),
        MockStreamDelta("is "),
        MockStreamDelta("a "),
        MockStreamDelta("subset "),
        MockStreamDelta("of "),
        MockStreamDelta("artificial "),
        MockStreamDelta("intelligence."),
    ]

    def get_mock_llm():
        return mock_llm_obj

    monkeypatch.setattr("app.services.query_service.get_llm", get_mock_llm)

    yield mock_llm_obj


@pytest.fixture
async def client(test_headers):
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers=test_headers,
    ) as ac:
        yield ac


@pytest.fixture
def sample_txt_content():
    """Sample text content for testing."""
    return """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

Key Concepts:
- Supervised Learning: Learning from labeled data
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through interaction with an environment

Applications:
Machine learning is used in various fields including:
- Natural language processing
- Computer vision
- Recommendation systems
- Predictive analytics
"""


@pytest.fixture
def sample_txt_file(sample_txt_content):
    """Create a temporary text file for testing."""
    import io

    return io.BytesIO(sample_txt_content.encode("utf-8"))