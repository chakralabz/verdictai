"""OpenAI-compatible hosted embedding provider."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from verdictai.ingestion.embeddings.config import (
    DEFAULT_HOSTED_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EmbeddingConfig,
)
from verdictai.ingestion.embeddings.embedder import Embedder
from verdictai.utils.errors import (
    EMBEDDING_API_KEY_MISSING,
    EMBEDDING_GENERATION_FAILED,
    EMBEDDING_RESPONSE_INVALID,
    EmbeddingError,
)


class OpenAIEmbeddingProvider(Embedder):
    """Generate embeddings through an OpenAI-compatible HTTP endpoint."""

    provider_name = "openai"

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the hosted embedding provider.

        Args:
            config: Provider configuration. Defaults to OpenAI embeddings.
        """

        if config is None:
            from verdictai.config import get_settings

            settings_config = get_settings().embeddings
        else:
            settings_config = config
        if settings_config.backend == "openai":
            self.config = settings_config
        elif settings_config.model_name == DEFAULT_LOCAL_EMBEDDING_MODEL:
            self.config = EmbeddingConfig(
                backend="openai",
                model_name=DEFAULT_HOSTED_EMBEDDING_MODEL,
                model_path=settings_config.model_path,
                model_cache_dir=settings_config.model_cache_dir,
                normalize_embeddings=settings_config.normalize_embeddings,
                batch_size=settings_config.batch_size,
                max_concurrency=settings_config.max_concurrency,
                request_timeout_seconds=settings_config.request_timeout_seconds,
                api_key=settings_config.api_key,
                endpoint_url=settings_config.endpoint_url,
                organization=settings_config.organization,
                dimensions=settings_config.dimensions,
            )
        elif config is None:
            self.config = EmbeddingConfig(
                backend="openai",
                model_name=DEFAULT_HOSTED_EMBEDDING_MODEL,
            )
        elif config.model_name == DEFAULT_LOCAL_EMBEDDING_MODEL:
            self.config = EmbeddingConfig(
                backend="openai",
                model_name=DEFAULT_HOSTED_EMBEDDING_MODEL,
                model_path=config.model_path,
                model_cache_dir=config.model_cache_dir,
                normalize_embeddings=config.normalize_embeddings,
                batch_size=config.batch_size,
                max_concurrency=config.max_concurrency,
                request_timeout_seconds=config.request_timeout_seconds,
                api_key=config.api_key,
                endpoint_url=config.endpoint_url,
                organization=config.organization,
                dimensions=config.dimensions,
            )
        else:
            self.config = config

    @property
    def model_name(self) -> str:
        """Return the hosted model name."""

        return self.config.model_name

    def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector."""

        return self.generate_batch_embeddings([text])[0]

    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate hosted embeddings in input order."""

        if not texts:
            return []
        response = self._post_embeddings(texts)
        return _extract_vectors(response, expected_count=len(texts))

    async def generate_embedding_async(self, text: str) -> list[float]:
        """Generate one hosted embedding without blocking the event loop."""

        return await asyncio.to_thread(self.generate_embedding, text)

    async def generate_batch_embeddings_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate hosted embeddings using bounded concurrent HTTP requests."""

        if not texts:
            return []
        batches = [
            texts[index : index + self.config.batch_size]
            for index in range(0, len(texts), self.config.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await asyncio.to_thread(self.generate_batch_embeddings, batch)

        results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        return [vector for batch in results for vector in batch]

    def _post_embeddings(self, texts: list[str]) -> dict[str, Any]:
        """Post one embeddings request to the configured endpoint."""

        api_key = self.config.api_key
        if not api_key:
            raise EmbeddingError(
                EMBEDDING_API_KEY_MISSING,
            )

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "input": texts,
        }
        if self.config.dimensions is not None:
            payload["dimensions"] = self.config.dimensions

        request = urllib.request.Request(
            self.config.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(
                EMBEDDING_GENERATION_FAILED,
                f"Hosted embedding request failed with HTTP {exc.code}: {detail}",
                provider=self.provider_name,
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise EmbeddingError(
                EMBEDDING_GENERATION_FAILED,
                provider=self.provider_name,
            ) from exc

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build request headers for an OpenAI-compatible endpoint."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization is not None:
            headers["OpenAI-Organization"] = self.config.organization
        return headers


def _extract_vectors(
    response: dict[str, Any],
    *,
    expected_count: int,
) -> list[list[float]]:
    """Extract ordered vectors from an OpenAI-compatible response."""

    data = response.get("data")
    if not isinstance(data, list):
        raise EmbeddingError(EMBEDDING_RESPONSE_INVALID, provider="openai")

    ordered = sorted(data, key=lambda item: item.get("index", 0))
    vectors: list[list[float]] = []
    for item in ordered:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list):
            raise EmbeddingError(EMBEDDING_RESPONSE_INVALID, provider="openai")
        vectors.append([float(value) for value in embedding])

    if len(vectors) != expected_count:
        raise EmbeddingError(EMBEDDING_RESPONSE_INVALID, provider="openai")
    return vectors
