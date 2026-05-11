"""Configuration models for Docling-backed chunkers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from verdictai.ingestion.chunker.docling.options.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TOKENIZER_PROVIDER,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class DoclingTokenizerConfig:
    """Configure tokenizer-backed Docling chunkers.

    Attributes:
        provider: Tokenizer provider aligned with the downstream embedding model.
        model_name: Provider-specific model or encoding name.
        max_tokens: Maximum token budget per contextualized chunk.
    """

    provider: Literal["huggingface", "openai"] = DEFAULT_TOKENIZER_PROVIDER
    model_name: str = DEFAULT_TOKENIZER_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(slots=True, frozen=True, kw_only=True)
class DoclingSerializerConfig:
    """Configure Docling chunk serialization behavior.

    Attributes:
        table_mode: Table serialization strategy used during chunking.
        compact_tables: Whether Markdown tables should use compact formatting.
        image_placeholder: Optional placeholder text for serialized pictures.
    """

    table_mode: Literal["default", "markdown"] = "default"
    compact_tables: bool = False
    image_placeholder: str | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class DoclingChunkerConfig:
    """Shared runtime configuration for Docling-backed chunkers.

    Attributes:
        tokenizer: Tokenizer configuration for token-aware chunkers.
        serializer: Serialization configuration passed to Docling when supported.
        merge_peers: Whether hybrid chunking should merge undersized peer chunks.
        repeat_table_header: Whether hybrid chunking should repeat table headers.
        omit_header_on_overflow: Whether hybrid chunking may omit headers on rows
            that only fit without them.
        prefix: Prefix repeated for line-based chunks.
        omit_prefix_on_overflow: Whether line-based chunking may omit the prefix on
            oversized lines that otherwise fit.
        merge_list_items: Whether hierarchical chunking should merge list items.
    """

    tokenizer: DoclingTokenizerConfig = field(default_factory=DoclingTokenizerConfig)
    serializer: DoclingSerializerConfig = field(default_factory=DoclingSerializerConfig)
    merge_peers: bool = True
    repeat_table_header: bool = True
    omit_header_on_overflow: bool = False
    prefix: str = ""
    omit_prefix_on_overflow: bool = False
    merge_list_items: bool = True
