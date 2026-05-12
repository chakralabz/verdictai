"""Pydantic schemas used by the embedding layer."""

from __future__ import annotations

from .embedded_chunk import EmbeddedChunk
from .embedding_result import EmbeddingResult

__all__ = ["EmbeddedChunk", "EmbeddingResult"]
