"""Configuration for embedding providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HOSTED_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

EmbeddingBackend = Literal["sentence_transformers", "openai"]


@dataclass(slots=True, frozen=True, kw_only=True)
class EmbeddingConfig:
    """Configure a concrete embedding provider.

    Attributes:
        backend: Provider backend used to create vectors.
        model_name: Provider model identifier. For local sentence-transformer
            models this may be a Hugging Face repo ID; for hosted OpenAI-compatible
            backends this is the remote model name.
        model_path: Optional path to a predownloaded local model directory.
        model_cache_dir: Optional cache root used by local model runtimes.
        normalize_embeddings: Whether local embeddings should be L2-normalized.
        batch_size: Maximum number of texts sent to one provider request.
        max_concurrency: Maximum concurrent hosted requests in async mode.
        request_timeout_seconds: Timeout for hosted embedding HTTP requests.
        max_retries: Number of SDK-level retries for transient hosted provider
            failures.
        api_key: Hosted provider API key. This may be supplied literally or
            through YAML environment interpolation such as `${OPENAI_API_KEY}`.
        endpoint_url: OpenAI-compatible embeddings endpoint.
        organization: Optional OpenAI organization header value.
        dimensions: Optional target dimension for hosted models that support it.
    """

    backend: EmbeddingBackend = "sentence_transformers"
    model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    model_path: Path | None = None
    model_cache_dir: Path | None = None
    normalize_embeddings: bool = True
    batch_size: int = 64
    max_concurrency: int = 4
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    api_key: str | None = None
    endpoint_url: str = DEFAULT_OPENAI_EMBEDDINGS_URL
    organization: str | None = None
    dimensions: int | None = None

    def resolved_model_reference(self) -> str:
        """Return the local model path when supplied, otherwise the model name.

        Returns:
            A filesystem path string or provider-specific model identifier.
        """

        if self.model_path is not None:
            return str(self.model_path.expanduser().resolve())
        return self.model_name

    def resolved_cache_dir(self) -> str | None:
        """Return the normalized local model cache directory when configured."""

        if self.model_cache_dir is None:
            return None
        return str(self.model_cache_dir.expanduser().resolve())
