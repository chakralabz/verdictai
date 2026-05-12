"""Embedding interfaces, schemas, and provider-backed implementations."""

from __future__ import annotations

from .config import EmbeddingBackend, EmbeddingConfig
from .embedder import Embedder, EmbedderProtocol
from .providers import OpenAIEmbeddingProvider, SentenceTransformerEmbeddingProvider
from .schemas import EmbeddedChunk, EmbeddingResult

__all__ = [
    "EmbeddedChunk",
    "Embedder",
    "EmbedderProtocol",
    "EmbeddingBackend",
    "EmbeddingConfig",
    "EmbeddingResult",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]
