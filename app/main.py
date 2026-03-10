"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import RequestIDMiddleware, setup_logging
from app.routers import health, ingest, query, stats
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)


async def ingest_startup_files():
    """
    Ingest files from DATA_DIR on startup (non-blocking).

    This runs in the background and logs results without blocking
    the application startup.
    """
    try:
        logger.info(f"Starting background ingestion from {settings.data_dir}")
        service = IngestService()
        result = await service.ingest_directory()
        logger.info(
            f"Startup ingestion complete: "
            f"{result.ingested} ingested, {result.skipped} skipped, "
            f"{len(result.errors)} errors"
        )
        if result.errors:
            for error in result.errors:
                logger.warning(f"Ingestion error: {error}")
    except Exception as e:
        logger.error(f"Error during startup ingestion: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    setup_logging()
    logger.info("Application starting up...")

    # Initialize Qdrant connection and embedding model
    try:
        from app.storage.qdrant_store import QdrantStore
        from app.services.ingest_service import get_embedding_model

        # Initialize Qdrant store (will create collection if needed)
        store = QdrantStore()
        logger.info("Qdrant connection initialized")

        # Initialize embedding model
        try:
            get_embedding_model()
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")

    except Exception as e:
        logger.error(f"Error initializing services: {e}", exc_info=True)

    # Start background ingestion task (non-blocking)
    asyncio.create_task(ingest_startup_files())

    yield

    # Shutdown
    logger.info("Application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG FastAPI Application",
    description="RAG system with LlamaIndex and Qdrant",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware (configurable via CORS_ORIGINS env var)
# Default: ["*"] (allow all origins)
# Production: Set CORS_ORIGINS to specific origins, e.g., "https://example.com,https://app.example.com"
cors_origins = settings.cors_origins
if cors_origins == ["*"]:
    logger.info("CORS configured to allow all origins (*)")
else:
    logger.info(f"CORS configured for origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(stats.router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "RAG FastAPI Application",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

