"""Service for querying the RAG system with LlamaIndex."""

import json
import logging
from typing import AsyncGenerator, Optional

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.core.prompts import PromptTemplate

from app.core.config import settings
from app.core.llm import get_llm, get_embed_model
from app.services.ingest_service import get_embedding_model
from app.storage.qdrant_store import QdrantStore, QdrantCollectionError

logger = logging.getLogger(__name__)


# RAG prompt template
RAG_PROMPT_TEMPLATE = PromptTemplate(
    """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the context does not contain enough information to answer the query, 
respond with: "I don't have enough information in the provided context to answer this question."

Query: {query_str}
Answer: """
)


class QueryService:
    """Service for querying documents in the RAG system."""

    def __init__(self):
        """Initialize the query service."""
        self.qdrant_store = QdrantStore()

    async def query_rag(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        stream: bool = False,
    ) -> dict:
        """
        Query the RAG system and generate an answer with sources.

        Args:
            query: The search query
            top_k: Number of results to return (will be clamped to TOP_K_MAX)
            filters: Optional metadata filters (e.g., {"filename": "doc.pdf"})
            stream: Whether to stream the response (not implemented yet)

        Returns:
            Dictionary with answer, sources, and retrieval_params

        Raises:
            ValueError: If query is empty or OpenAI API key is missing
            QdrantCollectionError: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Clamp top_k to TOP_K_MAX
        top_k = min(top_k, settings.top_k_max)

        # Get embedding model
        try:
            embedding_model = get_embedding_model()
        except Exception as e:
            raise ValueError(
                f"Failed to initialize embedding model: {str(e)}"
            ) from e

        # Generate query embedding
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = embedding_model.get_text_embedding(query)

        # Search in Qdrant with filters
        logger.info(f"Searching Qdrant with top_k={top_k}, filters={filters}")
        try:
            search_results = self.qdrant_store.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        except QdrantCollectionError as e:
            logger.error(f"Qdrant search failed: {e}")
            raise

        # Process results
        if not search_results:
            logger.info("No relevant documents found")
            return {
                "answer": (
                    "I don't have enough information in the provided context "
                    "to answer this question. No relevant documents were found."
                ),
                "sources": [],
                "retrieval_params": {
                    "top_k": top_k,
                    "filters": filters or {},
                },
            }

        # Extract context and build sources
        context_chunks = []
        sources = []

        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            score = result.get("score", 0.0)

            if not text:
                continue

            # Build source information
            source = {
                "doc_id": payload.get("document_id", ""),
                "filename": payload.get("filename", ""),
                "page": payload.get("page", None),  # May not be available for all docs
                "score": round(score, 4),
                "snippet": self._extract_snippet(text, query, max_length=200),
            }

            # Only include page if it exists
            if source["page"] is None:
                source.pop("page")

            sources.append(source)
            context_chunks.append(text)

        # Check if we have relevant context (score threshold)
        min_score = 0.5  # Minimum relevance score
        relevant_results = [s for s in sources if s["score"] >= min_score]

        if not relevant_results:
            logger.info("No results above relevance threshold")
            return {
                "answer": (
                    "I don't have enough information in the provided context "
                    "to answer this question. The retrieved documents are not "
                    "relevant enough to provide a reliable answer."
                ),
                "sources": sources,  # Still return sources even if not relevant
                "retrieval_params": {
                    "top_k": top_k,
                    "filters": filters or {},
                },
            }

        # Generate RAG response using LLM
        try:
            llm = get_llm()
            LlamaIndexSettings.llm = llm
        except Exception as e:
            # If LLM is not available, return context chunks without generation
            logger.warning(f"LLM not available: {e}. Returning context chunks only.")
            return {
                "answer": (
                    "LLM is not configured. Here are the relevant document chunks:\n\n"
                    + "\n\n---\n\n".join(context_chunks[:3])
                ),
                "sources": sources,
                "retrieval_params": {
                    "top_k": top_k,
                    "filters": filters or {},
                },
            }

        # Build context string from top chunks
        context_str = "\n\n---\n\n".join(context_chunks[:top_k])

        # Generate answer using RAG prompt
        logger.info("Generating RAG response with LLM")
        try:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context_str=context_str,
                query_str=query,
            )

            # Log query but not full context to avoid exposing document content
            logger.info(
                f"Query: {query[:100]}... | "
                f"Context chunks: {len(context_chunks)} | "
                f"Sources: {len(sources)}"
            )

            response = llm.complete(prompt)
            answer = str(response).strip()

            # Validate answer - if it's the "no info" response, return it
            if "don't have enough information" in answer.lower():
                return {
                    "answer": answer,
                    "sources": sources,
                    "retrieval_params": {
                        "top_k": top_k,
                        "filters": filters or {},
                    },
                }

            return {
                "answer": answer,
                "sources": sources,
                "retrieval_params": {
                    "top_k": top_k,
                    "filters": filters or {},
                },
            }

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}", exc_info=True)
            # Fallback: return context chunks
            return {
                "answer": (
                    f"Error generating response: {str(e)}. "
                    "Here are the relevant document chunks:\n\n"
                    + "\n\n---\n\n".join(context_chunks[:3])
                ),
                "sources": sources,
                "retrieval_params": {
                    "top_k": top_k,
                    "filters": filters or {},
                },
            }

    def _extract_snippet(
        self, text: str, query: str, max_length: int = 200
    ) -> str:
        """
        Extract a relevant snippet from text, preferably around query terms.

        Args:
            text: Full text to extract snippet from
            query: Query string to find relevant section
            max_length: Maximum length of snippet

        Returns:
            Snippet of text
        """
        if len(text) <= max_length:
            return text

        # Try to find query terms in text
        query_lower = query.lower()
        query_words = query_lower.split()

        # Find position of first query word
        best_pos = -1
        for word in query_words:
            if len(word) < 3:  # Skip very short words
                continue
            pos = text.lower().find(word)
            if pos != -1:
                best_pos = pos
                break

        if best_pos == -1:
            # No query terms found, return beginning
            return text[:max_length] + "..."

        # Extract snippet around query term
        start = max(0, best_pos - max_length // 2)
        end = min(len(text), start + max_length)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    async def query_rag_stream(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Query the RAG system and stream the answer with sources (SSE format).

        This method uses true token streaming when the LLM supports it (OpenAI).
        If the LLM doesn't support streaming, it falls back to pseudo-streaming
        by chunks of text (documented in the response).

        Args:
            query: The search query
            top_k: Number of results to return (will be clamped to TOP_K_MAX)
            filters: Optional metadata filters (e.g., {"filename": "doc.pdf"})

        Yields:
            SSE-formatted event strings

        Raises:
            ValueError: If query is empty or OpenAI API key is missing
            QdrantCollectionError: If search fails
        """
        if not query or not query.strip():
            yield self._sse_event("error", {"error": "Query cannot be empty"})
            return

        # Clamp top_k to TOP_K_MAX
        top_k = min(top_k, settings.top_k_max)

        # Get embedding model
        try:
            embedding_model = get_embedding_model()
        except Exception as e:
            yield self._sse_event(
                "error",
                {
                    "error": f"Failed to initialize embedding model: {str(e)}"
                },
            )
            return

        # Generate query embedding
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = embedding_model.get_text_embedding(query)

        # Search in Qdrant with filters
        logger.info(f"Searching Qdrant with top_k={top_k}, filters={filters}")
        try:
            search_results = self.qdrant_store.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        except QdrantCollectionError as e:
            logger.error(f"Qdrant search failed: {e}")
            yield self._sse_event("error", {"error": f"Search failed: {str(e)}"})
            return

        # Process results
        if not search_results:
            logger.info("No relevant documents found")
            yield self._sse_event(
                "answer",
                {
                    "text": (
                        "I don't have enough information in the provided context "
                        "to answer this question. No relevant documents were found."
                    ),
                    "done": True,
                },
            )
            yield self._sse_event("sources", {"sources": [], "done": True})
            return

        # Extract context and build sources
        context_chunks = []
        sources = []

        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            score = result.get("score", 0.0)

            if not text:
                continue

            # Build source information
            source = {
                "doc_id": payload.get("document_id", ""),
                "filename": payload.get("filename", ""),
                "page": payload.get("page", None),
                "score": round(score, 4),
                "snippet": self._extract_snippet(text, query, max_length=200),
            }

            # Only include page if it exists
            if source["page"] is None:
                source.pop("page", None)

            sources.append(source)
            context_chunks.append(text)

        # Check if we have relevant context (score threshold)
        min_score = 0.5
        relevant_results = [s for s in sources if s["score"] >= min_score]

        if not relevant_results:
            logger.info("No results above relevance threshold")
            yield self._sse_event(
                "answer",
                {
                    "text": (
                        "I don't have enough information in the provided context "
                        "to answer this question. The retrieved documents are not "
                        "relevant enough to provide a reliable answer."
                    ),
                    "done": True,
                },
            )
            yield self._sse_event("sources", {"sources": sources, "done": True})
            return

        # Generate RAG response using LLM with streaming
        try:
            llm = get_llm()
            LlamaIndexSettings.llm = llm
        except Exception as e:
            logger.warning(f"LLM not available: {e}. Using pseudo-streaming.")
            # Pseudo-streaming: send chunks of context
            answer_text = (
                "LLM is not configured. Here are the relevant document chunks:\n\n"
                + "\n\n---\n\n".join(context_chunks[:3])
            )
            # Split into chunks for pseudo-streaming
            chunk_size = 50  # Characters per chunk
            for i in range(0, len(answer_text), chunk_size):
                chunk = answer_text[i : i + chunk_size]
                yield self._sse_event(
                    "answer",
                    {"text": chunk, "done": False, "pseudo_stream": True},
                )
            yield self._sse_event("answer", {"text": "", "done": True})
            yield self._sse_event("sources", {"sources": sources, "done": True})
            return

        # Build context string
        context_str = "\n\n---\n\n".join(context_chunks[:top_k])

        # Generate answer using RAG prompt with streaming
        logger.info("Generating RAG response with LLM (streaming)")

        try:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context_str=context_str,
                query_str=query,
            )

            # Log query but not full context
            logger.info(
                f"Query: {query[:100]}... | "
                f"Context chunks: {len(context_chunks)} | "
                f"Sources: {len(sources)}"
            )

            # Stream response from LLM
            # OpenAI LLM supports streaming via stream_complete
            response_stream = llm.stream_complete(prompt)

            # Yield tokens as they arrive
            for token_delta in response_stream:
                token_text = str(token_delta.delta)
                if token_text:
                    yield self._sse_event(
                        "answer", {"text": token_text, "done": False}
                    )

            # Send final event
            yield self._sse_event("answer", {"text": "", "done": True})

            # Send sources event
            yield self._sse_event("sources", {"sources": sources, "done": True})

        except Exception as e:
            logger.error(f"Error generating streaming RAG response: {e}", exc_info=True)
            # Fallback: return error
            yield self._sse_event(
                "error",
                {
                    "error": f"Error generating response: {str(e)}",
                    "fallback": "\n\n---\n\n".join(context_chunks[:3]),
                },
            )
            yield self._sse_event("sources", {"sources": sources, "done": True})

    def _sse_event(self, event_type: str, data: dict) -> str:
        """
        Format data as Server-Sent Event (SSE).

        Args:
            event_type: Type of event (answer, sources, error)
            data: Event data dictionary

        Returns:
            SSE-formatted string
        """
        # Convert data to JSON
        data_json = json.dumps(data, ensure_ascii=False)

        # Format as SSE
        # Format: event: <type>\ndata: <json>\n\n
        return f"event: {event_type}\ndata: {data_json}\n\n"

    async def get_document(self, document_id: str) -> dict | None:
        """
        Get a specific document by ID.

        Args:
            document_id: SHA256 hash of the document

        Returns:
            Document data or None if not found
        """
        # TODO: Implement document retrieval by searching Qdrant for document_id
        return None
