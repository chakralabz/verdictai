"""Docling-backed chunker implementations."""

from __future__ import annotations

from .hierarchical_chunker import DoclingHierarchicalChunker
from .hybrid_chunker import DoclingHybridChunker
from .line_based_chunker import DoclingLineBasedChunker
from .options import (
    DoclingChunkerConfig,
    DoclingSerializerConfig,
    DoclingTokenizerConfig,
)

__all__ = [
    "DoclingChunkerConfig",
    "DoclingHierarchicalChunker",
    "DoclingHybridChunker",
    "DoclingLineBasedChunker",
    "DoclingSerializerConfig",
    "DoclingTokenizerConfig",
]


def __getattr__(name: str) -> object:
    """Import concrete chunkers only when callers request them."""

    if name == "DoclingHierarchicalChunker":
        from .hierarchical_chunker import DoclingHierarchicalChunker

        return DoclingHierarchicalChunker
    if name == "DoclingHybridChunker":
        from .hybrid_chunker import DoclingHybridChunker

        return DoclingHybridChunker
    if name == "DoclingLineBasedChunker":
        from .line_based_chunker import DoclingLineBasedChunker

        return DoclingLineBasedChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
