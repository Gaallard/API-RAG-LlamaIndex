"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app

transport = ASGITransport(app=app)

@pytest.mark.asyncio
async def test_health_ok():
    """Test health check endpoint returns 200 with correct structure."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        # Validate structure
        assert "status" in data
        assert data["status"] in ["ok", "degraded"]
        assert "vector_store" in data
        assert "reachable" in data["vector_store"]
        assert isinstance(data["vector_store"]["reachable"], bool)
        assert "collection" in data
        assert "name" in data["collection"]
        assert "points_count" in data["collection"]


@pytest.mark.asyncio
async def test_readiness_check():
    """Test readiness check endpoint."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ready", "not ready"]


@pytest.mark.asyncio
async def test_liveness_check():
    """Test liveness check endpoint."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "alive"
