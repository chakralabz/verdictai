"""Store protocol for retrieval-ready embedded chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from verdictai.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddingResult
from verdictai.ingestion.store.schemas import StoreResult


@runtime_checkable
class StoreProtocol(Protocol):
    """Define the storage behavior required by the ingestion pipeline.

    Implementations persist embedded chunks without exposing backend-specific
    concepts to parser, chunker, or embedding code. Stores must preserve the
    input order in their returned point identifiers and should make repeated
    writes of the same chunk IDs idempotent where the backend supports upsert.
    """

    def ensure_collection(self, *, vector_size: int) -> None:
        """Ensure the backend can store dense vectors of `vector_size`.

        Args:
            vector_size: Dimension of the dense embedding vector that will be
                stored for each chunk.
        """

    async def ensure_collection_async(self, *, vector_size: int) -> None:
        """Ensure the backend can store dense vectors without blocking."""

    def store_embeddings(
        self,
        embeddings: Sequence[EmbeddedChunk] | EmbeddingResult,
    ) -> StoreResult:
        """Persist embedded chunks.

        Args:
            embeddings: Either an `EmbeddingResult` or an ordered sequence of
                embedded chunks.

        Returns:
            Backend-neutral write metadata including stored point IDs.
        """

    async def store_embeddings_async(
        self,
        embeddings: Sequence[EmbeddedChunk] | EmbeddingResult,
    ) -> StoreResult:
        """Persist embedded chunks without blocking the event loop."""

    def close(self) -> None:
        """Release backend resources held by the store."""

    async def close_async(self) -> None:
        """Release backend resources without blocking the event loop."""


def normalize_embeddings(
    embeddings: Sequence[EmbeddedChunk] | EmbeddingResult,
) -> list[EmbeddedChunk]:
    """Return embedded chunks from any store input shape.

    Args:
        embeddings: Either an `EmbeddingResult` or an ordered sequence of
            embedded chunks.

    Returns:
        A concrete list of embedded chunks preserving input order.
    """

    if isinstance(embeddings, EmbeddingResult):
        return list(embeddings.chunks)
    return list(embeddings)
