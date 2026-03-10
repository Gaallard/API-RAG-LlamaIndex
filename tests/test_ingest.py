"""Tests for document ingestion endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_ingest_txt(client, sample_txt_file, mock_embeddings):
    """
    Test ingesting a small .txt file and validate ingested >= 1.
    """
    with patch("app.services.ingest_service.QdrantStore") as mock_qdrant_class:
        mock_store = MagicMock()
        mock_store.get_client.return_value = MagicMock()
        mock_store.collection_name = "test_collection"
        mock_store.check_connection.return_value = True

        async def mock_search_vectors(*args, **kwargs):
            return []

        async def mock_upsert_vectors(*args, **kwargs):
            return None

        async def mock_get_collection_stats():
            return {
                "collection_name": "test_collection",
                "total_documents": 0,
                "vector_dimension": 1536,
                "exists": True,
            }

        mock_store.search_vectors = mock_search_vectors
        mock_store.upsert_vectors = mock_upsert_vectors
        mock_store.get_collection_stats = mock_get_collection_stats
        mock_qdrant_class.return_value = mock_store

        files = [("files", ("test_ml.txt", sample_txt_file, "text/plain"))]
        response = await client.post("/ingest/", files=files)

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert "ingested" in data
        assert "skipped" in data
        assert "errors" in data
        assert data["ingested"] >= 1, (
            f"Expected at least 1 ingested file, got {data['ingested']}"
        )


@pytest.mark.asyncio
async def test_ingest_document_missing_api_key(sample_txt_file):
    """Test ingestion without API key."""
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [("files", ("test.txt", sample_txt_file, "text/plain"))]
        response = await client.post("/ingest/", files=files)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_document_invalid_api_key(client, sample_txt_file):
    """Test ingestion with invalid API key."""
    files = [("files", ("test.txt", sample_txt_file, "text/plain"))]
    headers = {"X-API-Key": "invalid-key"}
    response = await client.post("/ingest/", files=files, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_ingest_document_too_large(client, monkeypatch):
    """Test ingestion with file that's too large."""
    monkeypatch.setattr(settings, "file_max_mb", 1)

    file_content = b"x" * (2 * 1024 * 1024)
    files = [("files", ("large.txt", io.BytesIO(file_content), "text/plain"))]
    response = await client.post("/ingest/", files=files)

    assert response.status_code == 400