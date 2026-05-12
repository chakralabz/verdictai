"""Embedding provider protocol and base orchestration."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

from verdictai.ingestion.chunker.schemas import Chunk, ChunkingResult
from verdictai.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddingResult
from verdictai.ingestion.parser.types import JsonValue


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Define the embedding behavior required by the ingestion pipeline."""

    @property
    def provider_name(self) -> str:
        """Return the stable provider name used in metadata."""

    @property
    def model_name(self) -> str:
        """Return the configured embedding model identifier."""

    def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector."""

    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text in input order."""

    async def generate_embedding_async(self, text: str) -> list[float]:
        """Generate one embedding vector without blocking the event loop."""

    async def generate_batch_embeddings_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for text in input order asynchronously."""

    def embed_chunks(self, chunks: Sequence[Chunk] | ChunkingResult) -> EmbeddingResult:
        """Embed chunks emitted by the chunking layer."""

    async def embed_chunks_async(
        self,
        chunks: Sequence[Chunk] | ChunkingResult,
    ) -> EmbeddingResult:
        """Embed chunks emitted by the chunking layer asynchronously."""


class Embedder(ABC):
    """Base class for embedding providers.

    Concrete providers implement text-to-vector generation. This base class
    handles the ingestion-specific step of converting `Chunk` objects into
    `EmbeddedChunk` records while preserving provenance from chunking.
    """

    provider_name: str

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured embedding model identifier."""

    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector."""

    @abstractmethod
    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text in input order."""

    async def generate_embedding_async(self, text: str) -> list[float]:
        """Generate one embedding vector without blocking the event loop."""

        return await asyncio.to_thread(self.generate_embedding, text)

    async def generate_batch_embeddings_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for text in input order asynchronously."""

        return await asyncio.to_thread(self.generate_batch_embeddings, texts)

    def embed_chunks(self, chunks: Sequence[Chunk] | ChunkingResult) -> EmbeddingResult:
        """Embed chunks emitted by the chunking layer.

        Args:
            chunks: Either a `ChunkingResult` or an ordered sequence of chunks.

        Returns:
            Embedded chunks in input order plus provider metadata.
        """

        source_chunks, source_metadata = _normalize_chunks(chunks)
        texts = [_text_for_embedding(chunk) for chunk in source_chunks]
        vectors = self.generate_batch_embeddings(texts)
        embedded = _build_embedded_chunks(
            chunks=source_chunks,
            texts=texts,
            vectors=vectors,
            provider=self,
        )
        return EmbeddingResult(
            chunks=embedded,
            metadata=_result_metadata(
                source_metadata=source_metadata,
                provider=self,
                chunk_count=len(embedded),
            ),
        )

    async def embed_chunks_async(
        self,
        chunks: Sequence[Chunk] | ChunkingResult,
    ) -> EmbeddingResult:
        """Embed chunks emitted by the chunking layer asynchronously."""

        source_chunks, source_metadata = _normalize_chunks(chunks)
        texts = [_text_for_embedding(chunk) for chunk in source_chunks]
        vectors = await self.generate_batch_embeddings_async(texts)
        embedded = _build_embedded_chunks(
            chunks=source_chunks,
            texts=texts,
            vectors=vectors,
            provider=self,
        )
        return EmbeddingResult(
            chunks=embedded,
            metadata=_result_metadata(
                source_metadata=source_metadata,
                provider=self,
                chunk_count=len(embedded),
            ),
        )


def _normalize_chunks(
    chunks: Sequence[Chunk] | ChunkingResult,
) -> tuple[list[Chunk], dict[str, JsonValue]]:
    """Return chunk list plus upstream metadata."""

    if isinstance(chunks, ChunkingResult):
        return list(chunks.chunks), chunks.metadata
    return list(chunks), {}


def _text_for_embedding(chunk: Chunk) -> str:
    """Select the text representation used for embedding."""

    contextualized = chunk.contextualized_text.strip()
    if contextualized:
        return contextualized
    return chunk.text


def _build_embedded_chunks(
    *,
    chunks: list[Chunk],
    texts: list[str],
    vectors: list[list[float]],
    provider: EmbedderProtocol,
) -> list[EmbeddedChunk]:
    """Combine source chunks and provider vectors."""

    if len(vectors) != len(chunks):
        raise ValueError("Embedding provider returned a vector count mismatch.")

    return [
        EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            doc_id=chunk.doc_id,
            source_path=chunk.source_path,
            source_name=chunk.source_name,
            text=text,
            embedding=vector,
            embedding_model=provider.model_name,
            embedding_provider=provider.provider_name,
            metadata={
                "chunker_used": chunk.chunker_used,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "headings": cast(list[object], chunk.headings),
                "captions": cast(list[object], chunk.captions),
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "chunk_metadata": cast(dict[str, object], chunk.metadata),
            },
        )
        for chunk, text, vector in zip(chunks, texts, vectors, strict=True)
    ]


def _result_metadata(
    *,
    source_metadata: dict[str, JsonValue],
    provider: EmbedderProtocol,
    chunk_count: int,
) -> dict[str, JsonValue]:
    """Build request-level embedding metadata."""

    return {
        **source_metadata,
        "embedding_provider": provider.provider_name,
        "embedding_model": provider.model_name,
        "embedded_chunk_count": chunk_count,
    }
