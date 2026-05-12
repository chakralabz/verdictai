"""Qdrant implementation of the ingestion store interface."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from verdictai.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddingResult
from verdictai.ingestion.parser.types import JsonValue
from verdictai.ingestion.store.config import QdrantStoreConfig
from verdictai.ingestion.store.generic_store import normalize_embeddings
from verdictai.ingestion.store.schemas import StoreResult


class QdrantStore:
    """Store embedded chunks in Qdrant for dense plus BM25 hybrid retrieval.

    The collection uses two named vector fields:
    - `dense_vector_name` stores the embedding vector produced upstream.
    - `sparse_vector_name` stores a Qdrant FastEmbed BM25 document vector
      inferred from the same chunk text.

    Args:
        config: Optional Qdrant store configuration. When omitted, the process
            settings singleton is used.

    Raises:
        ImportError: If `qdrant-client[fastembed]` is not installed when a
            client operation is attempted.
    """

    backend_name = "qdrant"

    def __init__(self, config: QdrantStoreConfig | None = None) -> None:
        if config is None:
            from verdictai.config import get_settings

            config = get_settings().store.qdrant
        self.config = config
        self._client: Any | None = None
        self._async_client: Any | None = None

    def ensure_collection(self, *, vector_size: int) -> None:
        """Ensure the configured Qdrant collection exists."""

        _, client_cls, models = _import_qdrant()
        client = self._get_client(client_cls)
        if client.collection_exists(collection_name=self.config.collection_name):
            return
        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=self._dense_vectors_config(
                models=models,
                vector_size=vector_size,
            ),
            sparse_vectors_config=self._sparse_vectors_config(models=models),
        )

    async def ensure_collection_async(self, *, vector_size: int) -> None:
        """Ensure the configured Qdrant collection exists asynchronously."""

        async_client_cls, _, models = _import_qdrant()
        client = self._get_async_client(async_client_cls)
        exists = await client.collection_exists(
            collection_name=self.config.collection_name,
        )
        if exists:
            return
        await client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=self._dense_vectors_config(
                models=models,
                vector_size=vector_size,
            ),
            sparse_vectors_config=self._sparse_vectors_config(models=models),
        )

    def store_embeddings(
        self,
        embeddings: Sequence[EmbeddedChunk] | EmbeddingResult,
    ) -> StoreResult:
        """Store embedded chunks in Qdrant.

        Args:
            embeddings: Embedded chunks produced after chunk embedding.

        Returns:
            Result containing deterministic Qdrant point IDs.
        """

        chunks = normalize_embeddings(embeddings)
        if not chunks:
            return self._empty_result()

        _, client_cls, models = _import_qdrant()
        self._validate_embedding_dimensions(chunks)
        self.ensure_collection(vector_size=chunks[0].embedding_dimension)
        client = self._get_client(client_cls)
        point_ids = self._point_ids(chunks)

        for batch in _batched(chunks, self.config.batch_size):
            client.upsert(
                collection_name=self.config.collection_name,
                points=self._build_points(models=models, chunks=batch),
                wait=True,
            )

        return self._store_result(chunks=chunks, point_ids=point_ids)

    async def store_embeddings_async(
        self,
        embeddings: Sequence[EmbeddedChunk] | EmbeddingResult,
    ) -> StoreResult:
        """Store embedded chunks in Qdrant without blocking the event loop."""

        chunks = normalize_embeddings(embeddings)
        if not chunks:
            return self._empty_result()

        async_client_cls, _, models = _import_qdrant()
        self._validate_embedding_dimensions(chunks)
        await self.ensure_collection_async(vector_size=chunks[0].embedding_dimension)
        client = self._get_async_client(async_client_cls)
        point_ids = self._point_ids(chunks)

        for batch in _batched(chunks, self.config.batch_size):
            await client.upsert(
                collection_name=self.config.collection_name,
                points=self._build_points(models=models, chunks=batch),
                wait=True,
            )

        return self._store_result(chunks=chunks, point_ids=point_ids)

    def close(self) -> None:
        """Close the underlying synchronous Qdrant client when initialized."""

        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the underlying asynchronous Qdrant client when initialized."""

        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def _get_client(self, client_cls: Any) -> Any:
        """Return a lazily initialized synchronous Qdrant client."""

        if self._client is None:
            self._client = client_cls(**self._client_kwargs())
        return self._client

    def _get_async_client(self, async_client_cls: Any) -> Any:
        """Return a lazily initialized asynchronous Qdrant client."""

        if self._async_client is None:
            self._async_client = async_client_cls(**self._client_kwargs())
        return self._async_client

    def _client_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments shared by sync and async clients."""

        return {
            "url": self.config.url,
            "api_key": self.config.api_key or None,
            "prefer_grpc": self.config.prefer_grpc,
            "timeout": self.config.timeout_seconds,
        }

    def _dense_vectors_config(self, *, models: Any, vector_size: int) -> dict[str, Any]:
        """Build dense named-vector configuration."""

        return {
            self.config.dense_vector_name: models.VectorParams(
                size=vector_size,
                distance=_qdrant_distance(models=models, distance=self.config.distance),
                on_disk=self.config.dense_on_disk,
            )
        }

    def _sparse_vectors_config(self, *, models: Any) -> dict[str, Any]:
        """Build BM25 sparse named-vector configuration."""

        modifier = None
        if self.config.sparse_modifier == "idf":
            modifier = models.Modifier.IDF
        return {
            self.config.sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=self.config.sparse_on_disk),
                modifier=modifier,
            )
        }

    def _build_points(
        self, *, models: Any, chunks: Sequence[EmbeddedChunk]
    ) -> list[Any]:
        """Convert embedded chunks into Qdrant point structures."""

        return [
            models.PointStruct(
                id=_point_id(chunk),
                vector={
                    self.config.dense_vector_name: chunk.embedding,
                    self.config.sparse_vector_name: models.Document(
                        text=chunk.text,
                        model=self.config.sparse_model_name,
                    ),
                },
                payload=_payload(chunk),
            )
            for chunk in chunks
        ]

    def _validate_embedding_dimensions(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Ensure one write request does not mix embedding dimensions."""

        expected_dimension = chunks[0].embedding_dimension
        mismatches = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.embedding_dimension != expected_dimension
        ]
        if mismatches:
            raise ValueError(
                "Cannot store chunks with mixed embedding dimensions: "
                + ", ".join(mismatches)
            )

    def _point_ids(self, chunks: Sequence[EmbeddedChunk]) -> list[str]:
        """Return deterministic Qdrant point IDs for chunks."""

        return [_point_id(chunk) for chunk in chunks]

    def _empty_result(self) -> StoreResult:
        """Return a no-op write result."""

        return StoreResult(
            backend=self.backend_name,
            collection_name=self.config.collection_name,
            point_ids=[],
            stored_count=0,
            metadata={
                "dense_vector_name": self.config.dense_vector_name,
                "sparse_vector_name": self.config.sparse_vector_name,
                "sparse_model_name": self.config.sparse_model_name,
            },
        )

    def _store_result(
        self,
        *,
        chunks: Sequence[EmbeddedChunk],
        point_ids: list[str],
    ) -> StoreResult:
        """Build backend-neutral write metadata."""

        return StoreResult(
            backend=self.backend_name,
            collection_name=self.config.collection_name,
            point_ids=point_ids,
            stored_count=len(chunks),
            metadata={
                "dense_vector_name": self.config.dense_vector_name,
                "sparse_vector_name": self.config.sparse_vector_name,
                "sparse_model_name": self.config.sparse_model_name,
                "embedding_model": chunks[0].embedding_model,
                "embedding_provider": chunks[0].embedding_provider,
                "embedding_dimension": chunks[0].embedding_dimension,
            },
        )


def _import_qdrant() -> tuple[Any, Any, Any]:
    """Import Qdrant dependencies only when the store is used.

    Raises:
        ImportError: If the Qdrant client with FastEmbed support is unavailable.
    """

    try:
        from qdrant_client import AsyncQdrantClient, QdrantClient, models
    except ImportError as exc:
        raise ImportError(
            "Install `qdrant-client[fastembed]` to use QdrantStore."
        ) from exc
    return AsyncQdrantClient, QdrantClient, models


def _qdrant_distance(*, models: Any, distance: str) -> Any:
    """Map VerdictAI config values to Qdrant distance constants."""

    distances = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    return distances[distance]


def _point_id(chunk: EmbeddedChunk) -> str:
    """Return a deterministic UUID suitable for Qdrant point IDs."""

    return str(uuid5(NAMESPACE_URL, chunk.chunk_id))


def _payload(chunk: EmbeddedChunk) -> dict[str, JsonValue]:
    """Build JSON-safe Qdrant payload for one embedded chunk."""

    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "doc_id": chunk.doc_id,
        "source_path": chunk.source_path,
        "source_name": chunk.source_name,
        "text": chunk.text,
        "embedding_model": chunk.embedding_model,
        "embedding_provider": chunk.embedding_provider,
        "embedding_dimension": chunk.embedding_dimension,
        "metadata": cast(JsonValue, chunk.metadata),
    }


def _batched(
    chunks: Sequence[EmbeddedChunk],
    batch_size: int,
) -> Iterable[list[EmbeddedChunk]]:
    """Yield chunks in fixed-size batches preserving order."""

    for index in range(0, len(chunks), batch_size):
        yield list(chunks[index : index + batch_size])
