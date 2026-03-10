"""Query endpoints for RAG system."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import verify_api_key
from app.services.query_service import QueryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


class QueryFilters(BaseModel):
    """Metadata filters for query."""

    filename: Optional[str] = Field(
        default=None, description="Filter by filename (exact match)"
    )
    mime_type: Optional[str] = Field(
        default=None, description="Filter by MIME type (e.g., 'application/pdf')"
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"  # Don't allow extra fields


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    q: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(
        default=5,
        ge=1,
        le=settings.top_k_max,
        description=f"Number of results to return (max: {settings.top_k_max})",
    )
    filters: Optional[QueryFilters] = Field(
        default=None, description="Optional metadata filters"
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response using SSE"
    )


class Source(BaseModel):
    """Source document information."""

    doc_id: str = Field(..., description="Document ID (SHA256 hash)")
    filename: str = Field(..., description="Filename of the source document")
    page: Optional[int] = Field(
        default=None, description="Page number (if available)"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    snippet: str = Field(..., description="Snippet of the relevant text")


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    answer: str = Field(..., description="Generated answer based on context")
    sources: list[Source] = Field(
        default_factory=list, description="List of source documents"
    )
    retrieval_params: dict = Field(
        ..., description="Parameters used for retrieval (top_k, filters)"
    )


@router.post("/")
async def query_documents(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Query the RAG system and get an answer with cited sources.

    Supports both regular and streaming responses:
    - If stream=false (default): Returns JSON response
    - If stream=true: Returns SSE (Server-Sent Events) stream

    Args:
        request: Query request with query text, top_k, filters, and stream option
        api_key: Validated API key

    Returns:
        QueryResponse (JSON) or StreamingResponse (SSE) depending on stream parameter

    Raises:
        HTTPException: If query fails or OpenAI API key is missing
    """
    # Clamp top_k to TOP_K_MAX (already validated by Pydantic, but double-check)
    top_k = min(request.top_k, settings.top_k_max)

    # Build filters dictionary
    filters_dict = None
    if request.filters:
        filters_dict = {}
        if request.filters.filename:
            filters_dict["filename"] = request.filters.filename
        if request.filters.mime_type:
            filters_dict["mime_type"] = request.filters.mime_type

    # Log query (without exposing full content)
    logger.info(
        f"Query received: '{request.q[:100]}...' | "
        f"top_k={top_k} | filters={filters_dict} | stream={request.stream}"
    )

    service = QueryService()

    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            service.query_rag_stream(
                query=request.q,
                top_k=top_k,
                filters=filters_dict,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Handle regular (non-streaming) request
    try:
        result = await service.query_rag(
            query=request.q,
            top_k=top_k,
            filters=filters_dict,
            stream=False,
        )

        # Convert sources to Source models
        sources = [
            Source(
                doc_id=s["doc_id"],
                filename=s["filename"],
                page=s.get("page"),
                score=s["score"],
                snippet=s["snippet"],
            )
            for s in result["sources"]
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            retrieval_params=result["retrieval_params"],
        )

    except ValueError as e:
        # Missing OpenAI API key or invalid query
        logger.error(f"Query validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        ) from e


@router.get("/stream")
async def query_stream_get(
    q: str = Query(..., min_length=1, description="The search query"),
    top_k: int = Query(
        default=5, ge=1, le=settings.top_k_max, description="Number of results"
    ),
    filename: Optional[str] = Query(
        default=None, description="Filter by filename (exact match)"
    ),
    mime_type: Optional[str] = Query(
        default=None, description="Filter by MIME type"
    ),
    api_key: str = Depends(verify_api_key),
):
    """
    Query the RAG system with streaming response (SSE) via GET.

    This endpoint provides the same functionality as POST /query with stream=true,
    but uses query parameters for convenience.

    Args:
        q: The search query
        top_k: Number of results to return (max: TOP_K_MAX)
        filename: Optional filename filter
        mime_type: Optional MIME type filter
        api_key: Validated API key

    Returns:
        StreamingResponse with SSE events (text/event-stream)

    Raises:
        HTTPException: If query fails or OpenAI API key is missing
    """
    # Clamp top_k
    top_k = min(top_k, settings.top_k_max)

    # Build filters dictionary
    filters_dict = None
    if filename or mime_type:
        filters_dict = {}
        if filename:
            filters_dict["filename"] = filename
        if mime_type:
            filters_dict["mime_type"] = mime_type

    # Log query
    logger.info(
        f"Streaming query (GET): '{q[:100]}...' | "
        f"top_k={top_k} | filters={filters_dict}"
    )

    service = QueryService()

    return StreamingResponse(
        service.query_rag_stream(
            query=q,
            top_k=top_k,
            filters=filters_dict,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/similarity/{document_id}")
async def find_similar_documents(
    document_id: str,
    top_k: int = 5,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Find similar documents to a given document ID.

    Args:
        document_id: The document ID (SHA256) to find similar documents for
        top_k: Number of similar documents to return
        api_key: Validated API key

    Returns:
        Dictionary with similar documents

    Raises:
        HTTPException: If similarity search fails
    """
    # TODO: Implement similarity search
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Similarity search not yet implemented",
    )
