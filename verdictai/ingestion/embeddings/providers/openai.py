"""OpenAI-compatible hosted embedding provider."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
    """Generate embeddings through the official OpenAI Python SDK."""

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
                max_retries=settings_config.max_retries,
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
                max_retries=config.max_retries,
                api_key=config.api_key,
                endpoint_url=config.endpoint_url,
                organization=config.organization,
                dimensions=config.dimensions,
            )
        else:
            self.config = config
        self._client: Any | None = None
        self._async_client: Any | None = None

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
        try:
            response = self._get_client().embeddings.create(
                input=texts,
                model=self.config.model_name,
                **self._request_options(),
            )
        except _openai_api_error_types() as exc:
            raise EmbeddingError(
                EMBEDDING_GENERATION_FAILED,
                _sdk_error_detail(exc),
                provider=self.provider_name,
            ) from exc
        return _extract_vectors(response, expected_count=len(texts))

    async def generate_embedding_async(self, text: str) -> list[float]:
        """Generate one hosted embedding without blocking the event loop."""

        return await asyncio.to_thread(self.generate_embedding, text)

    async def generate_batch_embeddings_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate hosted embeddings using bounded concurrent SDK requests."""

        if not texts:
            return []
        batches = [
            texts[index : index + self.config.batch_size]
            for index in range(0, len(texts), self.config.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                try:
                    response = await self._get_async_client().embeddings.create(
                        input=batch,
                        model=self.config.model_name,
                        **self._request_options(),
                    )
                except _openai_api_error_types() as exc:
                    raise EmbeddingError(
                        EMBEDDING_GENERATION_FAILED,
                        _sdk_error_detail(exc),
                        provider=self.provider_name,
                    ) from exc
                return _extract_vectors(response, expected_count=len(batch))

        results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        return [vector for batch in results for vector in batch]

    def close(self) -> None:
        """Close the underlying synchronous SDK client when initialized."""

        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the underlying asynchronous SDK client when initialized."""

        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def _get_client(self) -> Any:
        """Return a lazily initialized OpenAI SDK client.

        Raises:
            EmbeddingError: If the hosted provider API key is not configured.
            ImportError: If the OpenAI SDK dependency is unavailable.
        """

        if self._client is None:
            kwargs = self._client_kwargs()
            _, client_cls, _ = _import_openai()
            self._client = client_cls(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        """Return a lazily initialized asynchronous OpenAI SDK client.

        Raises:
            EmbeddingError: If the hosted provider API key is not configured.
            ImportError: If the OpenAI SDK dependency is unavailable.
        """

        if self._async_client is None:
            kwargs = self._client_kwargs()
            _, _, async_client_cls = _import_openai()
            self._async_client = async_client_cls(**kwargs)
        return self._async_client

    def _client_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments shared by sync and async SDK clients."""

        api_key = self.config.api_key
        if not api_key:
            raise EmbeddingError(
                EMBEDDING_API_KEY_MISSING,
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": _base_url_from_endpoint_url(self.config.endpoint_url),
            "timeout": self.config.request_timeout_seconds,
            "max_retries": self.config.max_retries,
        }
        if self.config.organization is not None:
            kwargs["organization"] = self.config.organization
        return kwargs

    def _request_options(self) -> dict[str, int]:
        """Build per-request embedding options supported by hosted models."""

        if self.config.dimensions is None:
            return {}
        return {"dimensions": self.config.dimensions}


def _extract_vectors(
    response: Any,
    *,
    expected_count: int,
) -> list[list[float]]:
    """Extract ordered vectors from an OpenAI SDK embeddings response."""

    data = getattr(response, "data", None)
    if not isinstance(data, list):
        raise EmbeddingError(EMBEDDING_RESPONSE_INVALID, provider="openai")

    ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
    vectors: list[list[float]] = []
    for item in ordered:
        embedding = getattr(item, "embedding", None)
        if not isinstance(embedding, list):
            raise EmbeddingError(EMBEDDING_RESPONSE_INVALID, provider="openai")
        vectors.append([float(value) for value in embedding])

    if len(vectors) != expected_count:
        raise EmbeddingError(EMBEDDING_RESPONSE_INVALID, provider="openai")
    return vectors


def _import_openai() -> tuple[Any, Any, Any]:
    """Import OpenAI SDK dependencies only when hosted embeddings are used.

    Raises:
        ImportError: If the OpenAI Python SDK is unavailable.
    """

    try:
        import openai
        from openai import AsyncOpenAI, OpenAI
    except ImportError as exc:
        raise ImportError("Install `openai` to use OpenAIEmbeddingProvider.") from exc
    return openai, OpenAI, AsyncOpenAI


def _openai_api_error_types() -> tuple[type[Exception], ...]:
    """Return SDK exception classes that represent provider request failures."""

    openai, _, _ = _import_openai()
    return (openai.APIError,)


def _sdk_error_detail(exc: Exception) -> str:
    """Build a concise hosted embedding failure detail from an SDK exception."""

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        return str(exc)
    return f"Hosted embedding request failed with HTTP {status_code}: {exc}"


def _base_url_from_endpoint_url(endpoint_url: str) -> str:
    """Return the SDK base URL for an OpenAI-compatible embeddings endpoint.

    Args:
        endpoint_url: Either a base API URL such as `https://api.openai.com/v1`
            or the historical embeddings URL ending in `/embeddings`.

    Returns:
        Base URL accepted by the OpenAI SDK.
    """

    split_url = urlsplit(endpoint_url)
    path = split_url.path.rstrip("/")
    if path.endswith("/embeddings"):
        path = path.removesuffix("/embeddings") or "/"
    return urlunsplit(
        (
            split_url.scheme,
            split_url.netloc,
            path,
            split_url.query,
            split_url.fragment,
        )
    )
