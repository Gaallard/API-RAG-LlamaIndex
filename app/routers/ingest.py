"""Document ingestion endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import verify_api_key
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    """Response model for ingestion."""

    ingested: int
    skipped: int
    errors: list[dict[str, str]]


@router.post("/", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key),
) -> IngestResponse:
    """
    Ingest one or more documents into the RAG system.

    Args:
        files: List of document files to ingest (multipart/form-data)
        api_key: Validated API key

    Returns:
        IngestResponse with counts of ingested, skipped, and errors

    Raises:
        HTTPException: If no files provided or validation fails
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required",
        )

    # Validate file types and sizes
    files_data = []
    for file in files:
        if not file.filename:
            continue

        # Check file extension
        file_path = file.filename.lower()
        if not any(file_path.endswith(ext) for ext in settings.allowed_file_types):
            logger.warning(
                f"File '{file.filename}' has unsupported extension, skipping"
            )
            continue

        # Read file content
        try:
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)

            # Check file size
            if file_size_mb > settings.file_max_mb:
                logger.warning(
                    f"File '{file.filename}' ({file_size_mb:.2f}MB) exceeds "
                    f"maximum size ({settings.file_max_mb}MB), skipping"
                )
                continue

            # Determine MIME type
            mime_type: Optional[str] = file.content_type
            if mime_type is None:
                if file_path.endswith(".txt"):
                    mime_type = "text/plain"
                elif file_path.endswith(".pdf"):
                    mime_type = "application/pdf"

            files_data.append((file.filename, content, mime_type))

        except Exception as e:
            logger.error(f"Error reading file '{file.filename}': {e}")
            continue

    if not files_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files to ingest",
        )

    # Process files
    service = IngestService()
    try:
        result = await service.ingest_files(files_data)
        return IngestResponse(
            ingested=result.ingested,
            skipped=result.skipped,
            errors=result.errors,
        )
    except Exception as e:
        logger.error(f"Failed to ingest documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}",
        ) from e


@router.get("/status/{document_id}")
async def get_ingestion_status(
    document_id: str,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Get the ingestion status of a document.

    Args:
        document_id: The document ID (SHA256 hash) to check
        api_key: Validated API key

    Returns:
        Status information about the document
    """
    # TODO: Implement status check by querying Qdrant
    # For now, return basic response
    return {
        "document_id": document_id,
        "status": "unknown",
        "message": "Status check not yet fully implemented",
    }
