"""Health check endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.storage.qdrant_store import QdrantConnectionError, QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> dict:
    """
    Comprehensive health check endpoint.

    Returns:
        Dictionary with:
        - status: "ok" or "degraded"
        - vector_store: {"reachable": bool}
        - collection: {"name": str, "points_count": Optional[int]}
    """
    health_status = "ok"
    vector_store_reachable = False
    collection_info = {"name": "", "points_count": None}

    # Check Qdrant connection
    try:
        store = QdrantStore()
        vector_store_reachable = store.check_connection()

        if vector_store_reachable:
            # Try to get collection stats
            try:
                stats = await store.get_collection_stats()
                collection_info = {
                    "name": stats.get("collection_name", ""),
                    "points_count": stats.get("total_documents"),
                }
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
                health_status = "degraded"
        else:
            health_status = "degraded"

    except QdrantConnectionError as e:
        logger.warning(f"Qdrant connection error: {e}")
        health_status = "degraded"
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}", exc_info=True)
        health_status = "degraded"

    return {
        "status": health_status,
        "vector_store": {"reachable": vector_store_reachable},
        "collection": collection_info,
    }


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """
    Readiness check endpoint.

    Returns:
        Ready status message
    """
    # Check if Qdrant is reachable
    try:
        store = QdrantStore()
        if store.check_connection():
            return {"status": "ready"}
        else:
            return {"status": "not ready"}
    except Exception:
        return {"status": "not ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """
    Liveness check endpoint.

    Returns:
        Alive status message
    """
    return {"status": "alive"}
