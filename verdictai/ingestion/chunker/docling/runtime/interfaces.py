"""Internal interfaces for Docling chunker integration.

The public VerdictAI ingestion layer should only expose `DocumentParseResult`,
`ParsedBlock`, and `Chunk`. These protocols capture the small third-party
surface needed inside the Docling adapter so dynamic objects do not spread
through application code.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable


class DoclingDocumentProtocol(Protocol):
    """Marker protocol for a reconstructed Docling document."""


class DoclingChunkProtocol(Protocol):
    """Docling chunk shape consumed by the VerdictAI normalizer."""

    text: str
    meta: object


@runtime_checkable
class ChunkContextualizer(Protocol):
    """Docling chunker that can add heading/caption context to a chunk."""

    def contextualize(self, chunk: DoclingChunkProtocol) -> str:
        """Return chunk text enriched with contextual metadata."""


@runtime_checkable
class TokenCounter(Protocol):
    """Tokenizer that can count text tokens."""

    def count_tokens(self, text: str) -> int:
        """Count tokens for the supplied text."""


class DoclingChunkerProtocol(Protocol):
    """Chunker contract shared by Docling chunker implementations."""

    def chunk(
        self,
        dl_doc: DoclingDocumentProtocol,
    ) -> Iterable[DoclingChunkProtocol]:
        """Yield chunks for a reconstructed Docling document."""
