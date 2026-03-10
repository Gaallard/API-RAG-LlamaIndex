"""Qdrant vector store client and utilities."""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default vector dimensions for common embedding models
# These are fallback values if the embedding model cannot be detected
DEFAULT_VECTOR_DIMENSIONS = {
    "text-embedding-ada-002": 1536,  # OpenAI ada-002
    "text-embedding-3-small": 1536,  # OpenAI 3-small
    "text-embedding-3-large": 3072,  # OpenAI 3-large
    "nomic-embed-text": 768,  # Ollama nomic-embed-text
    "ollama": 768,  # Default Ollama embedding dimension (nomic-embed-text)
}


class QdrantStoreError(Exception):
    """Base exception for QdrantStore operations."""

    pass


class QdrantConnectionError(QdrantStoreError):
    """Exception raised when connection to Qdrant fails."""

    pass


class QdrantCollectionError(QdrantStoreError):
    """Exception raised when collection operations fail."""

    pass


class QdrantStore:
    """Qdrant vector store client wrapper with persistent connection."""

    def __init__(self, vector_size: Optional[int] = None):
        """
        Initialize Qdrant client and ensure collection exists.

        Args:
            vector_size: Optional vector size. If not provided, will attempt to
                        detect from embedding model or use fallback.

        Raises:
            QdrantConnectionError: If connection to Qdrant fails
        """
        self.collection_name = settings.qdrant_collection
        self._vector_size = vector_size
        self._client: Optional[QdrantClient] = None

        try:
            self._client = QdrantClient(url=settings.qdrant_url)
            # Test connection
            self._client.get_collections()
            logger.info(
                f"Connected to Qdrant at {self._sanitize_url(settings.qdrant_url)}"
            )
        except Exception as e:
            error_msg = f"Failed to connect to Qdrant at {self._sanitize_url(settings.qdrant_url)}"
            logger.error(f"{error_msg}: {str(e)}")
            raise QdrantConnectionError(error_msg) from e

        # Ensure collection exists (will use detected or fallback vector size)
        if vector_size is None:
            detected_size = self._detect_vector_size()
            if detected_size:
                self._vector_size = detected_size
                logger.info(f"Detected vector size: {detected_size}")
            else:
                # Use fallback
                self._vector_size = self._get_fallback_vector_size()
                logger.warning(
                    f"Could not detect vector size, using fallback: {self._vector_size}"
                )

        self.ensure_collection(self._vector_size)

    def _sanitize_url(self, url: str) -> str:
        """
        Sanitize URL for logging (remove credentials if present).

        Args:
            url: URL to sanitize

        Returns:
            Sanitized URL without credentials
        """
        # Remove any potential credentials from URL
        if "@" in url:
            parts = url.split("@")
            if len(parts) == 2:
                return f"***@{parts[1]}"
        return url

    def _detect_vector_size(self) -> Optional[int]:
        """
        Attempt to detect vector size from embedding model.

        This method tries to detect the embedding model being used by LlamaIndex
        and returns its vector dimension. If detection fails, returns None.

        Detection methods (in order of preference):
        1. Infer from embedding provider and model name (PRIORITY - model config overrides existing collection)
        2. Try to infer from OpenAI API key presence (assumes ada-002)
        3. Check if collection already exists and get its vector size (only if model not detected)
        4. Return None to use fallback

        Returns:
            Vector size if detected, None otherwise
        """
        # Method 1: Infer from embedding provider and model name (PRIORITY)
        # This ensures we use the correct dimension based on current configuration
        if settings.embed_provider.lower() == "ollama":
            model_name = settings.ollama_embed_model.lower()
            if model_name in DEFAULT_VECTOR_DIMENSIONS:
                detected_size = DEFAULT_VECTOR_DIMENSIONS[model_name]
                logger.info(f"Detected Ollama model '{model_name}', using dimension {detected_size}")
                return detected_size
            # Default Ollama dimension
            detected_size = DEFAULT_VECTOR_DIMENSIONS["ollama"]
            logger.info(f"Using default Ollama embedding dimension ({detected_size})")
            return detected_size
        
        # Method 2: Infer from OpenAI API key (if present, likely using OpenAI embeddings)
        if settings.openai_api_key:
            model_name = settings.openai_embed_model.lower()
            if model_name in DEFAULT_VECTOR_DIMENSIONS:
                detected_size = DEFAULT_VECTOR_DIMENSIONS[model_name]
                logger.info(f"Detected OpenAI model '{model_name}', using dimension {detected_size}")
                return detected_size
            detected_size = DEFAULT_VECTOR_DIMENSIONS["text-embedding-ada-002"]
            logger.info(f"OpenAI API key detected, assuming text-embedding-ada-002 ({detected_size})")
            return detected_size

        # Method 3: Check if collection exists and get its vector size (fallback only)
        # Only use this if we couldn't detect from model configuration
        try:
            if self._client:
                collection_info = self._client.get_collection(self.collection_name)
                if collection_info:
                    vector_size = collection_info.config.params.vectors.size
                    logger.warning(
                        f"Could not detect model dimension, using existing collection size: {vector_size}"
                    )
                    return vector_size
        except Exception as e:
            logger.debug(f"Could not get existing collection info: {str(e)}")

        # Could not detect
        return None

    def _get_fallback_vector_size(self) -> int:
        """
        Get fallback vector size based on configuration.

        Returns:
            Default vector size based on embed_provider setting
        """
        # Use provider-specific default
        if settings.embed_provider.lower() == "ollama":
            return DEFAULT_VECTOR_DIMENSIONS["ollama"]
        # Default to OpenAI ada-002 dimension
        return DEFAULT_VECTOR_DIMENSIONS["text-embedding-ada-002"]

    def get_client(self) -> QdrantClient:
        """
        Get the Qdrant client instance.

        Returns:
            QdrantClient instance

        Raises:
            QdrantConnectionError: If client is not initialized
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        """
        Ensure the collection exists, create it if it doesn't.

        Args:
            vector_size: Size of vectors to store in the collection

        Raises:
            QdrantCollectionError: If collection creation fails
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")

        try:
            # Check if collection exists using collection_exists method if available
            # Otherwise fall back to checking collections list
            collection_exists = False
            try:
                # Try using collection_exists if available (newer API)
                if hasattr(self._client, "collection_exists"):
                    collection_exists = self._client.collection_exists(
                        self.collection_name
                    )
                else:
                    # Fallback: check collections list
                    collections = self._client.get_collections().collections
                    collection_names = [col.name for col in collections]
                    collection_exists = self.collection_name in collection_names
            except Exception as e:
                logger.debug(f"Could not check collection existence: {str(e)}")
                # Assume it doesn't exist and try to create

            if collection_exists:
                # Verify vector size matches
                try:
                    collection_info = self._client.get_collection(
                        self.collection_name
                    )
                    existing_size = collection_info.config.params.vectors.size
                    points_count = collection_info.points_count
                    
                    if existing_size != vector_size:
                        # If collection is empty, delete and recreate it
                        if points_count == 0:
                            logger.warning(
                                f"Collection '{self.collection_name}' has wrong vector size "
                                f"({existing_size} vs required {vector_size}) and is empty. "
                                f"Deleting and recreating with correct size."
                            )
                            try:
                                self._client.delete_collection(self.collection_name)
                                logger.info(f"Deleted collection '{self.collection_name}'")
                                # Continue to create new collection below
                            except Exception as e:
                                error_msg = (
                                    f"Failed to delete collection '{self.collection_name}': {str(e)}"
                                )
                                logger.error(error_msg)
                                raise QdrantCollectionError(error_msg) from e
                        else:
                            # Collection has data, cannot auto-fix
                            error_msg = (
                                f"Collection '{self.collection_name}' exists with "
                                f"vector size {existing_size}, but required size is "
                                f"{vector_size}. Collection has {points_count} points. "
                                f"This mismatch will cause errors. "
                                f"Please delete the collection manually or change the embedding model. "
                                f"To delete: curl -X DELETE http://localhost:6333/collections/{self.collection_name} "
                                f"or use: docker exec rag-qdrant qdrant-cli collection delete {self.collection_name}"
                            )
                            logger.error(error_msg)
                            raise QdrantCollectionError(error_msg)
                    else:
                        logger.info(f"Collection '{self.collection_name}' already exists with correct vector size ({vector_size})")
                        return
                except QdrantCollectionError:
                    raise
                except Exception as e:
                    logger.warning(
                        f"Could not verify collection vector size: {str(e)}"
                    )
                    # If we can't verify, assume it's OK and continue
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return

            # Create collection
            logger.info(
                f"Creating collection '{self.collection_name}' with vector size {vector_size}"
            )
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")

        except Exception as e:
            error_msg = f"Failed to ensure collection '{self.collection_name}': {str(e)}"
            logger.error(error_msg)
            raise QdrantCollectionError(error_msg) from e

    async def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics:
            - collection_name: Name of the collection
            - total_documents: Number of points in the collection
            - vector_dimension: Vector dimension size
            - exists: Whether the collection exists

        Raises:
            QdrantConnectionError: If client is not initialized
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")

        try:
            collection_info = self._client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_documents": collection_info.points_count,
                "vector_dimension": collection_info.config.params.vectors.size,
                "exists": True,
            }
        except Exception as e:
            logger.warning(
                f"Could not get collection stats for '{self.collection_name}': {str(e)}"
            )
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "vector_dimension": None,
                "exists": False,
            }

    async def get_lightweight_stats(self) -> dict:
        """
        Get lightweight statistics about the collection (for /stats endpoint).

        Uses efficient queries to avoid performance issues.

        Returns:
            Dictionary with:
            - chunks_count: Total number of chunks (points)
            - documents_count: Number of unique documents
            - last_ingest_timestamp: Most recent ingest timestamp (ISO format)
            - collection_size_estimate: Estimated size in bytes (if available)

        Raises:
            QdrantConnectionError: If client is not initialized
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")

        try:
            # Get collection info (lightweight)
            collection_info = self._client.get_collection(self.collection_name)
            chunks_count = collection_info.points_count

            # Estimate collection size (rough calculation)
            # Vector size * 4 bytes (float32) + payload estimate
            vector_dim = collection_info.config.params.vectors.size
            vector_size_per_point = vector_dim * 4  # float32 = 4 bytes
            payload_estimate = 1000  # Rough estimate per payload in bytes
            collection_size_estimate = chunks_count * (vector_size_per_point + payload_estimate)

            # Get unique document count and last ingest timestamp
            # Use scroll with limit to sample efficiently
            documents_set = set()
            last_ingest_timestamp = None

            # Scroll through points in batches (limit to avoid heavy queries)
            scroll_limit = 1000  # Sample up to 1000 points for efficiency
            scroll_result = self._client.scroll(
                collection_name=self.collection_name,
                limit=scroll_limit,
                with_payload=True,
                with_vectors=False,  # Don't fetch vectors to save bandwidth
            )

            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                payload = point.payload or {}
                doc_id = payload.get("document_id")
                if doc_id:
                    documents_set.add(doc_id)

                # Track most recent ingest timestamp
                ingest_ts = payload.get("ingest_timestamp")
                if ingest_ts:
                    if last_ingest_timestamp is None or ingest_ts > last_ingest_timestamp:
                        last_ingest_timestamp = ingest_ts

            # If we sampled less than total, estimate document count
            documents_count = len(documents_set)
            if chunks_count > scroll_limit:
                # Estimate: assume average chunks per document from sample
                if documents_count > 0:
                    avg_chunks_per_doc = scroll_limit / documents_count
                    documents_count = int(chunks_count / avg_chunks_per_doc)
                else:
                    documents_count = 0

            return {
                "chunks_count": chunks_count,
                "documents_count": documents_count,
                "last_ingest_timestamp": last_ingest_timestamp,
                "collection_size_estimate": collection_size_estimate,
            }

        except Exception as e:
            logger.warning(f"Could not get lightweight stats: {str(e)}")
            return {
                "chunks_count": 0,
                "documents_count": 0,
                "last_ingest_timestamp": None,
                "collection_size_estimate": 0,
            }

    def check_connection(self) -> bool:
        """
        Check if Qdrant connection is reachable.

        Returns:
            True if connection is reachable, False otherwise
        """
        if self._client is None:
            return False

        try:
            # Lightweight check: try to get collections list
            self._client.get_collections()
            return True
        except Exception:
            return False

    async def upsert_vectors(
        self, points: list[dict]
    ) -> None:  # type: ignore[type-arg]
        """
        Upsert vectors into the collection.

        Args:
            points: List of point dictionaries with id, vector, and payload

        Raises:
            QdrantCollectionError: If upsertion fails
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")

        try:
            from qdrant_client.models import PointStruct

            # Validate and convert point IDs to ensure they're proper types
            point_structs = []
            for point in points:
                point_id = point["id"]
                
                # Ensure ID is a native Python int (not numpy.int64, etc.)
                if not isinstance(point_id, int):
                    # Try to convert if it's a numeric type
                    try:
                        point_id = int(point_id)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Point ID must be an integer or convertible to int, "
                            f"got {type(point_id)}: {point_id}"
                        ) from e
                
                # Validate ID is in valid range for uint64
                if point_id < 0 or point_id >= 2**64:
                    raise ValueError(
                        f"Point ID must be in range [0, 2^64), got: {point_id}"
                    )
                
                point_structs.append(
                    PointStruct(
                        id=point_id,
                        vector=point["vector"],
                        payload=point.get("payload", {}),
                    )
                )

            self._client.upsert(
                collection_name=self.collection_name, points=point_structs
            )
            logger.info(f"Upserted {len(points)} points to collection")
        except Exception as e:
            error_msg = f"Failed to upsert vectors: {str(e)}"
            logger.error(error_msg)
            raise QdrantCollectionError(error_msg) from e

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar vectors with optional filters.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional dictionary with metadata filters (e.g., {"filename": "doc.pdf"})

        Returns:
            List of search results with id, score, and payload

        Raises:
            QdrantCollectionError: If search fails
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Build query filter if filters are provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if value is not None:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value),
                            )
                        )

                if conditions:
                    query_filter = Filter(must=conditions)

            # Perform search using query_points (new qdrant-client API)
            search_params = {
                "collection_name": self.collection_name,
                "query": query_vector,
                "limit": top_k,
                "with_payload": True,
            }
            if query_filter:
                search_params["query_filter"] = query_filter

            res = self._client.query_points(**search_params)

            return [
                {
                    "id": str(p.id),
                    "score": float(p.score) if p.score is not None else 0.0,
                    "payload": p.payload or {},
                }
                for p in res.points
            ]

            return [
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload or {},
                }
                for r in results
            ]
        except Exception as e:
            error_msg = f"Failed to search vectors: {str(e)}"
            logger.error(error_msg)
            raise QdrantCollectionError(error_msg) from e

    async def delete_vectors(self, point_ids: list[str]) -> None:
        """
        Delete vectors by IDs.

        Args:
            point_ids: List of point IDs to delete

        Raises:
            QdrantCollectionError: If deletion fails
        """
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialized")

        try:
            from qdrant_client.models import PointIdsList

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points from collection")
        except Exception as e:
            error_msg = f"Failed to delete vectors: {str(e)}"
            logger.error(error_msg)
            raise QdrantCollectionError(error_msg) from e

