from app.core.config import settings

# OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Ollama (optional - only imported if needed)
try:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    Ollama = None  # type: ignore
    OllamaEmbedding = None  # type: ignore


def get_llm():
    if settings.llm_provider.lower() == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama provider requested but llama-index-llms-ollama is not installed. "
                "Install it with: pip install llama-index-llms-ollama"
            )
        return Ollama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
        )

    # default: OpenAI
    return OpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )


def get_embed_model():
    if settings.embed_provider.lower() == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama provider requested but llama-index-embeddings-ollama is not installed. "
                "Install it with: pip install llama-index-embeddings-ollama"
            )
        return OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    # default: OpenAI
    return OpenAIEmbedding(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
