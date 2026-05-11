"""Runtime contract for ingestion-facing document chunkers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verdictai.ingestion.chunker.schemas import ChunkingResult
from verdictai.ingestion.parser.schemas import DocumentParseResult


@runtime_checkable
class DocumentChunkerProtocol(Protocol):
    """Define the chunking capabilities required by the ingestion pipeline."""

    def chunk(self, parse_result: DocumentParseResult) -> ChunkingResult:
        """Chunk a parsed document into retrieval-ready segments.

        Args:
            parse_result: Structured parser output to chunk.

        Returns:
            A `ChunkingResult` containing emitted chunks plus metadata.
        """

    async def chunk_async(self, parse_result: DocumentParseResult) -> ChunkingResult:
        """Chunk a parsed document asynchronously.

        Args:
            parse_result: Structured parser output to chunk.

        Returns:
            A `ChunkingResult` containing emitted chunks plus metadata.
        """
