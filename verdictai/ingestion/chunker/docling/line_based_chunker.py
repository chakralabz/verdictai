"""Line-preserving Docling chunker implementation."""

from __future__ import annotations

from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker

from verdictai.ingestion.chunker.docling.options import (
    DOCLING_LINE_BASED_CHUNKER_NAME,
)
from verdictai.ingestion.chunker.docling.base_docling_chunker import BaseDoclingChunker
from verdictai.ingestion.chunker.docling.runtime.interfaces import (
    DoclingChunkerProtocol,
)
from verdictai.ingestion.chunker.docling.runtime.tokenizers import (
    build_docling_tokenizer,
)
from verdictai.ingestion.parser.types import JsonValue


class DoclingLineBasedChunker(BaseDoclingChunker):
    """Use Docling's `LineBasedTokenChunker` on serialized parser output."""

    chunker_name = DOCLING_LINE_BASED_CHUNKER_NAME

    def _create_docling_chunker(self) -> DoclingChunkerProtocol:
        """Create a configured Docling `LineBasedTokenChunker`."""

        return LineBasedTokenChunker(
            tokenizer=build_docling_tokenizer(config=self.config.tokenizer),
            prefix=self.config.prefix,
            omit_prefix_on_overflow=self.config.omit_prefix_on_overflow,
        )

    def _build_result_metadata(self) -> dict[str, JsonValue]:
        """Return metadata specific to line-based token chunking."""

        return {
            "prefix": self.config.prefix,
            "omit_prefix_on_overflow": self.config.omit_prefix_on_overflow,
            "tokenizer_provider": self.config.tokenizer.provider,
            "tokenizer_model": self.config.tokenizer.model_name,
            "max_tokens": self.config.tokenizer.max_tokens,
        }
