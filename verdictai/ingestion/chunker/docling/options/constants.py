"""Constants for Docling-backed chunkers."""

from __future__ import annotations

from typing import Final, Literal

DEFAULT_TOKENIZER_PROVIDER: Final[Literal["huggingface", "openai"]] = "huggingface"
DEFAULT_TOKENIZER_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_TOKENS: Final[int] = 512

DOCLING_HYBRID_CHUNKER_NAME: Final[str] = "docling_hybrid"
DOCLING_HIERARCHICAL_CHUNKER_NAME: Final[str] = "docling_hierarchical"
DOCLING_LINE_BASED_CHUNKER_NAME: Final[str] = "docling_line_based"
