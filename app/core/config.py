"""Application configuration using pydantic-settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    api_key: str = Field(
        default="dev-api-key-change-in-production",
        description="API key for authentication (required in production, set via API_KEY env var)",
    )
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (optional if using Ollama)"
    )

        # Provider selection
    llm_provider: str = Field(
        default="ollama", description="LLM provider: 'openai' or 'ollama'"
    )
    embed_provider: str = Field(
        default="ollama", description="Embeddings provider: 'openai' or 'ollama'"
    )

    # OpenAI models (used when provider is openai)
    openai_llm_model: str = Field(
        default="gpt-4o-mini", description="OpenAI chat model name"
    )
    openai_embed_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model name"
    )

    # Ollama configuration (used when provider is ollama)
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        description="Ollama base URL (from Docker, use host.docker.internal)",
    )
    ollama_llm_model: str = Field(
        default="llama3.1:8b", description="Ollama LLM model name"
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text", description="Ollama embeddings model name"
    )

    # CORS Configuration
    # Can be set as comma-separated string in env var: CORS_ORIGINS=https://example.com,https://app.example.com
    # Or as list in code. Default: ["*"] (allow all)
    cors_origins: str | list[str] = Field(
        default="*",
        description="CORS origins: '*' for all, or comma-separated list of origins",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_collection: str = Field(
        default="rag_documents", description="Qdrant collection name"
    )

    # Application Settings
    data_dir: Path = Field(
        default=Path("./data"), description="Directory for storing uploaded files"
    )
    top_k_max: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )
    file_max_mb: int = Field(
        default=10, ge=1, le=100, description="Maximum file size in MB"
    )

    # Chunking Settings
    chunk_size: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="Size of text chunks in characters (default: 1024)",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=512,
        description="Overlap between chunks in characters (default: 200)",
    )

    # Allowed file types
    allowed_file_types: list[str] = Field(
        default_factory=lambda: [".txt", ".pdf"],
        description="List of allowed file extensions",
    )

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    def __init__(self, **kwargs):
        """Initialize settings and create data directory if it doesn't exist."""
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Warn if using default API key
        if self.api_key == "dev-api-key-change-in-production":
            import warnings

            warnings.warn(
                "Using default API key. Set API_KEY in .env file for production!",
                UserWarning,
            )

        # Parse CORS origins from env var if it's a string
        # Support both comma-separated string and list
        if isinstance(self.cors_origins, str):
            if self.cors_origins == "*":
                self.cors_origins = ["*"]
            else:
                self.cors_origins = [
                    origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
                ]
        elif not isinstance(self.cors_origins, list):
            # Fallback to default
            self.cors_origins = ["*"]


settings = Settings()

