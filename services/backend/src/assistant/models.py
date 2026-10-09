from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings

from assistant.config import settings

# Per request; a hung call would otherwise hold a worker slot for the SDK's 10 minutes.
TIMEOUT_SECONDS = 30


def chat_model(name: str, **kwargs: Any) -> BaseChatModel:
    """Chat model on the configured OpenAI-compatible API (OpenAI or OpenRouter)."""
    return init_chat_model(
        name,
        model_provider="openai",
        base_url=settings.openai_base_url,
        timeout=TIMEOUT_SECONDS,
        **kwargs,
    )


def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.openai_base_url,
        timeout=TIMEOUT_SECONDS,
        # Non-OpenAI APIs take text, not the token ids LangChain sends by default.
        check_embedding_ctx_length=settings.openai_base_url is None,
    )
