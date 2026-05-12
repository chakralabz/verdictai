"""SentenceTransformers-backed local embedding provider."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from verdictai.ingestion.embeddings.config import EmbeddingConfig
from verdictai.ingestion.embeddings.embedder import Embedder
from verdictai.utils.errors import (
    EMBEDDING_DEPENDENCY_MISSING,
    EMBEDDING_GENERATION_FAILED,
    EMBEDDING_MODEL_LOAD_FAILED,
    EmbeddingError,
)


class SentenceTransformerEmbeddingProvider(Embedder):
    """Generate embeddings with a local SentenceTransformers model.

    Notes:
        The provider accepts either a Hugging Face model ID or `model_path` for
        predownloaded model directories. The optional `sentence_transformers`
        dependency is imported lazily so hosted-only deployments can still import
        the embedding package.
    """

    provider_name = "sentence_transformers"

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the provider without loading the model immediately.

        Args:
            config: Provider configuration. Defaults to local sentence-transformer
                settings.
        """

        if config is not None:
            self.config = config
        else:
            from verdictai.config import get_settings

            settings_config = get_settings().embeddings
            self.config = (
                settings_config
                if settings_config.backend == self.provider_name
                else EmbeddingConfig()
            )
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        """Return the configured model reference."""

        return self.config.resolved_model_reference()

    def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector."""

        return self.generate_batch_embeddings([text])[0]

    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings locally in input order."""

        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=False,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise EmbeddingError(
                EMBEDDING_GENERATION_FAILED,
                provider=self.provider_name,
            ) from exc
        return [_coerce_vector(vector) for vector in vectors]

    async def generate_embedding_async(self, text: str) -> list[float]:
        """Generate one local embedding without blocking the event loop."""

        return await asyncio.to_thread(self.generate_embedding, text)

    async def generate_batch_embeddings_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate local embeddings without blocking the event loop."""

        return await asyncio.to_thread(self.generate_batch_embeddings, texts)

    def _load_model(self) -> Any:
        """Load and cache the SentenceTransformer model instance.

        Raises:
            EmbeddingError: If the optional runtime or model is unavailable.
        """

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                EMBEDDING_DEPENDENCY_MISSING,
                dependency="sentence-transformers",
                backend=self.provider_name,
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.config.resolved_model_reference(),
                cache_folder=self.config.resolved_cache_dir(),
            )
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise EmbeddingError(
                EMBEDDING_MODEL_LOAD_FAILED,
                model_name=self.model_name,
            ) from exc
        return self._model


def _coerce_vector(vector: Sequence[float] | Any) -> list[float]:
    """Convert provider vector output into plain Python floats."""

    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
