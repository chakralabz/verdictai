"""Hybrid Docling chunker implementation."""

from __future__ import annotations

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from verdictai.ingestion.chunker.docling.base_docling_chunker import BaseDoclingChunker
from verdictai.ingestion.chunker.docling.options import DOCLING_HYBRID_CHUNKER_NAME
from verdictai.ingestion.chunker.docling.runtime.interfaces import (
    DoclingChunkerProtocol,
)
from verdictai.ingestion.chunker.docling.runtime.serializers import (
    build_serializer_provider,
)
from verdictai.ingestion.chunker.docling.runtime.tokenizers import (
    build_docling_tokenizer,
)
from verdictai.ingestion.parser.types import JsonValue


class DoclingHybridChunker(BaseDoclingChunker):
    """Use Docling's `HybridChunker` on serialized parser output."""

    chunker_name = DOCLING_HYBRID_CHUNKER_NAME

    def _create_docling_chunker(self) -> DoclingChunkerProtocol:
        """Create a configured Docling `HybridChunker`."""

        serializer_provider = build_serializer_provider(config=self.config.serializer)
        if serializer_provider is not None:
            return HybridChunker(
                tokenizer=build_docling_tokenizer(config=self.config.tokenizer),
                merge_peers=self.config.merge_peers,
                repeat_table_header=self.config.repeat_table_header,
                omit_header_on_overflow=self.config.omit_header_on_overflow,
                serializer_provider=serializer_provider,
            )
        return HybridChunker(
            tokenizer=build_docling_tokenizer(config=self.config.tokenizer),
            merge_peers=self.config.merge_peers,
            repeat_table_header=self.config.repeat_table_header,
            omit_header_on_overflow=self.config.omit_header_on_overflow,
        )

    def _build_result_metadata(self) -> dict[str, JsonValue]:
        """Return metadata specific to hybrid chunking."""

        return {
            "merge_peers": self.config.merge_peers,
            "repeat_table_header": self.config.repeat_table_header,
            "omit_header_on_overflow": self.config.omit_header_on_overflow,
            "tokenizer_provider": self.config.tokenizer.provider,
            "tokenizer_model": self.config.tokenizer.model_name,
            "max_tokens": self.config.tokenizer.max_tokens,
            "table_mode": self.config.serializer.table_mode,
        }
