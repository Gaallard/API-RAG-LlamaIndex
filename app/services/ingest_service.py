"""Service for document ingestion with LlamaIndex and Qdrant."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from llama_index.core import Document, Settings as LlamaIndexSettings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import FlatReader, PDFReader

from app.core.llm import get_embed_model
from app.core.config import settings
from app.storage.qdrant_store import QdrantStore, QdrantCollectionError

logger = logging.getLogger(__name__)

# Initialize embedding model
# Initialize embedding model (provider-agnostic)
_embedding_model: Optional[BaseEmbedding] = None


def get_embedding_model() -> BaseEmbedding:
    """
    Get or create the embedding model instance (OpenAI or Ollama based on settings).

    Returns:
        BaseEmbedding instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embed_model()  # <--- usa tu selector (ollama/openai)
        # Set as global embedding model for LlamaIndex
        LlamaIndexSettings.embed_model = _embedding_model
        logger.info(
            f"Embedding model initialized via provider='{getattr(settings, 'embed_provider', 'unknown')}'"
        )
    return _embedding_model



class DocumentMetadata:
    """Metadata for a document."""

    def __init__(
        self,
        filename: str,
        mime_type: str,
        size: int,
        sha256: str,
        ingest_timestamp: datetime,
    ):
        """Initialize document metadata."""
        self.filename = filename
        self.mime_type = mime_type
        self.size = size
        self.sha256 = sha256
        self.ingest_timestamp = ingest_timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "sha256": self.sha256,
            "ingest_timestamp": self.ingest_timestamp.isoformat(),
        }


class IngestResult:
    """Result of ingestion operation."""

    def __init__(self):
        """Initialize ingestion result counters."""
        self.ingested = 0
        self.skipped = 0
        self.errors: list[dict[str, str]] = []

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "ingested": self.ingested,
            "skipped": self.skipped,
            "errors": self.errors,
        }


class IngestService:
    """Service for ingesting documents into the RAG system."""

    def __init__(self):
        """Initialize the ingestion service."""
        self.qdrant_store = QdrantStore()
        self.data_dir = settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        try:
            embedding_model = get_embedding_model()
            # Get vector size - OpenAI embeddings typically have a known dimension
            # For text-embedding-ada-002 it's 1536, for text-embedding-3-small it's 1536, etc.
            # We'll let QdrantStore handle vector size detection
            logger.info(f"Embedding model initialized via provider='{settings.embed_provider}'")

        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            # Will use default vector size from QdrantStore

        # Initialize chunking
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info(
            f"Chunking configured: size={settings.chunk_size}, "
            f"overlap={settings.chunk_overlap}"
        )

    def _get_file_reader(self, file_path: Path):
        """
        Get appropriate reader for file type.

        Args:
            file_path: Path to the file

        Returns:
            Reader instance (FlatReader or PDFReader)

        Raises:
            ValueError: If file type is not supported
        """
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return FlatReader()
        elif suffix == ".pdf":
            return PDFReader()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _calculate_sha256(self, content: bytes) -> str:
        """
        Calculate SHA256 hash of content.

        Args:
            content: File content as bytes

        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(content).hexdigest()

    def _check_duplicate(self, sha256: str) -> bool:
        """
        Check if document with given SHA256 already exists in Qdrant.

        Args:
            sha256: SHA256 hash of the document

        Returns:
            True if document exists, False otherwise
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = self.qdrant_store.get_client()

            # Search for any point with this sha256 in payload
            # We use a dummy query vector (all zeros) since we're filtering by payload
            # Get vector dimension from collection
            collection_info = client.get_collection(self.qdrant_store.collection_name)
            vector_dim = collection_info.config.params.vectors.size
            dummy_vector = [0.0] * vector_dim

            # Search with filter for sha256
            results = client.search(
                collection_name=self.qdrant_store.collection_name,
                query_vector=dummy_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="sha256",
                            match=MatchValue(value=sha256),
                        )
                    ]
                ),
                limit=1,
            )

            # If we found any results, document exists
            return len(results) > 0

        except Exception as e:
            logger.debug(f"Could not check duplicate for {sha256[:8]}...: {e}")
            # On error, assume not duplicate to allow ingestion
            return False

    async def _process_document(
        self, file_path: Path, content: bytes, metadata: DocumentMetadata
    ) -> tuple[list[Document], int]:
        """
        Process document: load, chunk, and generate embeddings.

        Args:
            file_path: Path to the file
            content: File content as bytes
            metadata: Document metadata

        Returns:
            Tuple of (list of Document nodes, number of chunks)

        Raises:
            Exception: If processing fails
        """
        # Load document using appropriate reader
        reader = self._get_file_reader(file_path)
        documents = reader.load_data(file_path)

        if not documents:
            raise ValueError(f"No content extracted from {file_path}")

        # Add metadata to documents
        for doc in documents:
            doc.metadata.update(metadata.to_dict())
            doc.metadata["document_id"] = metadata.sha256

        # Split into chunks
        nodes = self.text_splitter.get_nodes_from_documents(documents)
        logger.info(
            f"Document '{metadata.filename}' split into {len(nodes)} chunks"
        )

        return nodes, len(nodes)

    async def _store_in_qdrant(
        self, nodes: list[Document], metadata: DocumentMetadata
    ) -> None:
        """
        Store document chunks in Qdrant with embeddings.

        Args:
            nodes: List of document nodes (chunks)
            metadata: Document metadata

        Raises:
            QdrantCollectionError: If storage fails
        """
        embedding_model = get_embedding_model()
        points = []

        for idx, node in enumerate(nodes):
            # Generate embedding (OpenAIEmbedding.get_text_embedding is sync)
            embedding = embedding_model.get_text_embedding(node.text)
            # Ensure embedding is a list of floats
            if not isinstance(embedding, list):
                raise ValueError(f"Expected list of floats, got {type(embedding)}")

            # Generate deterministic point ID compatible with Qdrant
            # Qdrant accepts: unsigned integers (uint64) or UUIDs (as strings)
            # We use a hash-based integer ID for simplicity and compatibility
            # This ensures the same document+chunk always gets the same ID
            # Combine SHA256 and chunk index to create a unique integer ID
            combined = f"{metadata.sha256}_{idx}".encode("utf-8")
            # Use hash to generate a deterministic integer (modulo to fit in uint64)
            hash_int = int(hashlib.sha256(combined).hexdigest()[:16], 16)
            # Ensure it fits in uint64 range (0 to 2^64 - 1) and is a native Python int
            point_id = int(hash_int % (2**64))
            
            # Validate ID type (must be int, not string or other type)
            if not isinstance(point_id, int):
                raise TypeError(f"Point ID must be int, got {type(point_id)}: {point_id}")

            point = {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "document_id": metadata.sha256,
                    "sha256": metadata.sha256,  # For deduplication check
                    "chunk_index": idx,
                    "text": node.text,
                    "filename": metadata.filename,
                    "mime_type": metadata.mime_type,
                    "size": metadata.size,
                    "ingest_timestamp": metadata.ingest_timestamp.isoformat(),
                    **node.metadata,
                },
            }
            points.append(point)

        # Upsert to Qdrant
        await self.qdrant_store.upsert_vectors(points)
        logger.info(
            f"Stored {len(points)} chunks for document '{metadata.filename}'"
        )

    async def ingest_file(
        self, filename: str, content: bytes, mime_type: Optional[str] = None
    ) -> dict[str, str]:
        """
        Ingest a single file into the RAG system.

        Args:
            filename: Name of the file
            content: File content as bytes
            mime_type: Optional MIME type of the file

        Returns:
            Dictionary with document_id and status

        Raises:
            ValueError: If file type is not supported or file is invalid
            Exception: If ingestion fails
        """
        # Validate file type
        file_path = Path(filename)
        suffix = file_path.suffix.lower()
        if suffix not in settings.allowed_file_types:
            raise ValueError(
                f"File type {suffix} not allowed. "
                f"Allowed types: {', '.join(settings.allowed_file_types)}"
            )

        # Calculate SHA256 for deduplication
        sha256 = self._calculate_sha256(content)

        # Check for duplicates
        if self._check_duplicate(sha256):
            logger.info(f"Document with SHA256 {sha256[:8]}... already exists, skipping")
            return {
                "document_id": sha256,
                "status": "skipped",
                "reason": "duplicate",
            }

        # Create metadata
        if mime_type is None:
            mime_type = "text/plain" if suffix == ".txt" else "application/pdf"

        metadata = DocumentMetadata(
            filename=filename,
            mime_type=mime_type,
            size=len(content),
            sha256=sha256,
            ingest_timestamp=datetime.utcnow(),
        )

        # Save file to data directory
        file_path = self.data_dir / filename
        file_path.write_bytes(content)

        # Process document: load, chunk, embed
        nodes, num_chunks = await self._process_document(
            file_path, content, metadata
        )

        # Store in Qdrant
        await self._store_in_qdrant(nodes, metadata)

        logger.info(
            f"Successfully ingested '{filename}' "
            f"(SHA256: {sha256[:8]}..., {num_chunks} chunks)"
        )

        return {
            "document_id": sha256,
            "status": "ingested",
            "chunks": num_chunks,
        }

    async def ingest_files(
        self, files: list[tuple[str, bytes, Optional[str]]]
    ) -> IngestResult:
        """
        Ingest multiple files.

        Args:
            files: List of tuples (filename, content, mime_type)

        Returns:
            IngestResult with counts and errors
        """
        result = IngestResult()

        for filename, content, mime_type in files:
            try:
                # Validate file size
                file_size_mb = len(content) / (1024 * 1024)
                if file_size_mb > settings.file_max_mb:
                    result.errors.append(
                        {
                            "filename": filename,
                            "error": f"File size ({file_size_mb:.2f}MB) exceeds maximum ({settings.file_max_mb}MB)",
                        }
                    )
                    continue

                # Ingest file
                ingest_result = await self.ingest_file(filename, content, mime_type)

                if ingest_result["status"] == "skipped":
                    result.skipped += 1
                else:
                    result.ingested += 1

            except Exception as e:
                logger.error(f"Error ingesting '{filename}': {e}", exc_info=True)
                result.errors.append(
                    {"filename": filename, "error": str(e)}
                )

        return result

    async def ingest_directory(self, directory: Optional[Path] = None) -> IngestResult:
        """
        Ingest all .txt and .pdf files from a directory.

        Args:
            directory: Directory to ingest from (defaults to DATA_DIR)

        Returns:
            IngestResult with counts and errors
        """
        if directory is None:
            directory = self.data_dir

        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return IngestResult()

        files_to_ingest = []
        for suffix in settings.allowed_file_types:
            files_to_ingest.extend(directory.glob(f"*{suffix}"))

        if not files_to_ingest:
            logger.info(f"No files found in {directory}")
            return IngestResult()

        logger.info(f"Found {len(files_to_ingest)} files to ingest in {directory}")

        files_data = []
        for file_path in files_to_ingest:
            try:
                content = file_path.read_bytes()
                mime_type = (
                    "text/plain" if file_path.suffix == ".txt" else "application/pdf"
                )
                files_data.append((file_path.name, content, mime_type))
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        return await self.ingest_files(files_data)

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the RAG system.

        Args:
            document_id: SHA256 hash of the document to delete

        Returns:
            True if deleted successfully
        """
        # TODO: Implement document deletion
        # Need to:
        # 1. Find all points with document_id in payload
        # 2. Delete them from Qdrant
        # 3. Optionally delete file from data_dir
        return False
