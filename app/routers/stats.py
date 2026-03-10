"""Statistics endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import verify_api_key
from app.storage.qdrant_store import QdrantConnectionError, QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


class ProvidersInfo(BaseModel):
    """Provider information."""

    llm: Optional[str] = Field(
        default=None, description="LLM provider (e.g., 'openai', 'ollama')"
    )
    embeddings: Optional[str] = Field(
        default=None, description="Embeddings provider (e.g., 'openai', 'ollama')"
    )


class StatsResponse(BaseModel):
    """Response model for statistics."""

    documents_count: int = Field(..., description="Number of unique documents")
    chunks_count: int = Field(..., description="Total number of chunks (vectors)")
    last_ingest_timestamp: Optional[str] = Field(
        default=None, description="Most recent ingest timestamp (ISO format)"
    )
    collection_size_estimate: int = Field(
        default=0, description="Estimated collection size in bytes"
    )
    providers: ProvidersInfo = Field(..., description="LLM and embeddings providers")


@router.get("/", response_model=StatsResponse)
async def get_stats(
    api_key: str = Depends(verify_api_key),
) -> StatsResponse:
    """
    Get detailed statistics about the RAG system.

    Uses lightweight queries to Qdrant to avoid performance issues.

    Args:
        api_key: Validated API key

    Returns:
        StatsResponse with system statistics

    Raises:
        HTTPException: If Qdrant connection fails
    """
    try:
        store = QdrantStore()

        # Get lightweight stats from Qdrant
        lightweight_stats = await store.get_lightweight_stats()

        # Determine providers from config
        llm_provider = None
        embeddings_provider = None

        if settings.openai_api_key:
            # Check if LLM is configured (OpenAI)
            try:
                from app.services.query_service import get_llm

                llm = get_llm()
                llm_provider = "openai"
            except (ValueError, Exception):
                llm_provider = None

            # Embeddings provider
            try:
                from app.services.ingest_service import get_embedding_model

                embedding_model = get_embedding_model()
                embeddings_provider = "openai"
            except (ValueError, Exception):
                embeddings_provider = None

        # If no OpenAI, could be Ollama (not implemented yet)
        # For now, we only support OpenAI

        return StatsResponse(
            documents_count=lightweight_stats.get("documents_count", 0),
            chunks_count=lightweight_stats.get("chunks_count", 0),
            last_ingest_timestamp=lightweight_stats.get("last_ingest_timestamp"),
            collection_size_estimate=lightweight_stats.get(
                "collection_size_estimate", 0
            ),
            providers=ProvidersInfo(
                llm=llm_provider,
                embeddings=embeddings_provider,
            ),
        )

    except QdrantConnectionError as e:
        logger.error(f"Qdrant connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant connection error: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        # Return default stats on error
        return StatsResponse(
            documents_count=0,
            chunks_count=0,
            last_ingest_timestamp=None,
            collection_size_estimate=0,
            providers=ProvidersInfo(llm=None, embeddings=None),
        )
