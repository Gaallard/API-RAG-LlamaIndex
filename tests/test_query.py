"""Tests for query endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_query(client, sample_txt_content, mock_embeddings, mock_llm):
    """
    Test query endpoint after ingesting a document.
    """
    with patch("app.services.ingest_service.QdrantStore") as mock_ingest_qdrant, \
         patch("app.services.query_service.QdrantStore") as mock_query_qdrant:

        mock_store = MagicMock()
        mock_store.get_client.return_value = MagicMock()
        mock_store.collection_name = "test_collection"
        mock_store.check_connection.return_value = True

        def mock_search_vectors(*args, **kwargs):
            return [
                {
                    "id": "test_id_1",
                    "score": 0.85,
                    "payload": {
                        "document_id": "test_doc_123",
                        "filename": "test_ml.txt",
                        "text": sample_txt_content[:500],
                        "chunk_index": 0,
                        "mime_type": "text/plain",
                        "size": len(sample_txt_content.encode()),
                        "ingest_timestamp": "2024-01-15T10:30:00",
                    },
                }
            ]

        async def mock_upsert_vectors(*args, **kwargs):
            return None

        async def mock_get_collection_stats():
            return {
                "collection_name": "test_collection",
                "total_documents": 1,
                "vector_dimension": 1536,
                "exists": True,
            }

        mock_store.search_vectors = mock_search_vectors
        mock_store.upsert_vectors = mock_upsert_vectors
        mock_store.get_collection_stats = mock_get_collection_stats

        mock_ingest_qdrant.return_value = mock_store
        mock_query_qdrant.return_value = mock_store

        sample_file = io.BytesIO(sample_txt_content.encode("utf-8"))
        files = [("files", ("test_ml.txt", sample_file, "text/plain"))]
        ingest_response = await client.post("/ingest/", files=files)

        assert ingest_response.status_code == 200, (
            f"Expected 200, got {ingest_response.status_code}: {ingest_response.text}"
        )

        ingest_data = ingest_response.json()
        assert ingest_data["ingested"] >= 1

        query_response = await client.post(
            "/query/",
            json={
                "q": "What is machine learning?",
                "top_k": 5,
                "stream": False,
            },
        )

        assert query_response.status_code == 200, (
            f"Expected 200, got {query_response.status_code}: {query_response.text}"
        )

        query_data = query_response.json()

        assert "answer" in query_data
        assert "sources" in query_data
        assert "retrieval_params" in query_data

        assert query_data["answer"], "Answer should not be empty"
        assert len(query_data["answer"].strip()) > 0

        assert len(query_data["sources"]) >= 1, (
            f"Expected at least 1 source, got {len(query_data['sources'])}"
        )
        assert query_data["sources"][0]["doc_id"]
        assert query_data["sources"][0]["filename"]
        assert query_data["sources"][0]["score"] > 0


@pytest.mark.asyncio
async def test_query_missing_api_key():
    """Test query without API key."""
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query/",
            json={"q": "test query", "top_k": 5},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_invalid_api_key(client):
    """Test query with invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    response = await client.post(
        "/query/",
        json={"q": "test query", "top_k": 5},
        headers=headers,
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_invalid_top_k(client, monkeypatch):
    """Test query with invalid top_k value."""
    monkeypatch.setattr(settings, "top_k_max", 10)

    response = await client.post(
        "/query/",
        json={"q": "test query", "top_k": 100},
    )
    assert response.status_code == 422