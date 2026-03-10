"""Security utilities for API key authentication."""

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key", description="API key for authentication")
) -> str:
    """
    Dependency to verify API key from X-API-Key header.

    This dependency is required for protected endpoints:
    - /ingest/*
    - /query/*
    - /stats/*

    Health endpoints (/health/*) are public and do not require authentication.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The API key if valid

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key

