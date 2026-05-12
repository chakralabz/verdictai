"""Chunking interfaces, schemas, and Docling-backed implementations."""

from __future__ import annotations


from .docling import (
    DoclingChunkerConfig,
    DoclingSerializerConfig,
    DoclingTokenizerConfig,
)
from .document_chunker import DocumentChunkerProtocol
from .schemas import Chunk, ChunkingResult
from .docling import (
    DoclingHierarchicalChunker,
    DoclingHybridChunker,
    DoclingLineBasedChunker,
)

__all__ = [
    "Chunk",
    "ChunkingResult",
    "DoclingChunkerConfig",
    "DoclingHierarchicalChunker",
    "DoclingHybridChunker",
    "DoclingLineBasedChunker",
    "DoclingSerializerConfig",
    "DoclingTokenizerConfig",
    "DocumentChunkerProtocol",
]


def __getattr__(name: str) -> object:
    """Import concrete Docling chunkers only when requested."""

    if name in {
        "DoclingHierarchicalChunker",
        "DoclingHybridChunker",
        "DoclingLineBasedChunker",
    }:
        from . import docling

        return getattr(docling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
