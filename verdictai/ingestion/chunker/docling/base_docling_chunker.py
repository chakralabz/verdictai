"""Shared base implementation for Docling-backed chunkers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from verdictai.ingestion.chunker.docling.options import DoclingChunkerConfig
from verdictai.ingestion.chunker.docling.runtime.interfaces import (
    DoclingChunkerProtocol,
    DoclingChunkProtocol,
    DoclingDocumentProtocol,
)
from verdictai.ingestion.chunker.docling.runtime.normalize_docling_chunks import (
    normalize_docling_chunks,
)
from verdictai.ingestion.chunker.docling.runtime.restore_docling_document import (
    restore_docling_document,
)
from verdictai.ingestion.chunker.document_chunker import DocumentChunkerProtocol
from verdictai.ingestion.chunker.schemas import ChunkingResult
from verdictai.ingestion.parser.schemas import DocumentParseResult
from verdictai.ingestion.parser.types import JsonValue


class BaseDoclingChunker(DocumentChunkerProtocol, ABC):
    """Adapt a Docling chunker to VerdictAI's chunking interface."""

    chunker_name = "docling"

    def __init__(self, config: DoclingChunkerConfig | None = None) -> None:
        """Create a Docling-backed chunker adapter.

        Args:
            config: Chunker configuration. When omitted, defaults are used.
        """

        if config is None:
            from verdictai.config import get_settings

            self.config = get_settings().chunker.docling
        else:
            self.config = config

    def chunk(self, parse_result: DocumentParseResult) -> ChunkingResult:
        """Chunk parser output into retrieval-ready VerdictAI chunks."""

        # 1. Reconstruct the Docling document from parser output so we can
        #    use Docling's chunkers directly instead of chunking flattened text.
        document = restore_docling_document(parse_result=parse_result)

        # 2. Create the concrete chunker implementation chosen by the subclass.
        docling_chunker = self._create_docling_chunker()

        # 3. Run Docling chunking and normalize the result to VerdictAI schemas.
        docling_chunks = self._chunk_document(
            docling_chunker=docling_chunker,
            document=document,
        )
        chunks = normalize_docling_chunks(
            chunker_name=self.chunker_name,
            docling_chunker=docling_chunker,
            docling_chunks=docling_chunks,
            parse_result=parse_result,
        )

        metadata: dict[str, JsonValue] = {
            "chunker_used": self.chunker_name,
            "chunk_count": len(chunks),
            "source_name": parse_result.metadata.get("source_name"),
        }
        metadata.update(self._build_result_metadata())
        return ChunkingResult(chunks=chunks, metadata=metadata)

    async def chunk_async(self, parse_result: DocumentParseResult) -> ChunkingResult:
        """Chunk parser output asynchronously using a worker thread."""

        return await asyncio.to_thread(self.chunk, parse_result)

    def _chunk_document(
        self,
        *,
        docling_chunker: DoclingChunkerProtocol,
        document: DoclingDocumentProtocol,
    ) -> list[DoclingChunkProtocol]:
        """Run a Docling chunker against a reconstructed document."""

        return list(docling_chunker.chunk(dl_doc=document))

    @abstractmethod
    def _create_docling_chunker(self) -> DoclingChunkerProtocol:
        """Create the underlying Docling chunker."""

    @abstractmethod
    def _build_result_metadata(self) -> dict[str, JsonValue]:
        """Return chunker-specific result metadata."""
