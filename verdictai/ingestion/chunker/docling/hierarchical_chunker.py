"""Structure-aware Docling chunker implementation."""

from __future__ import annotations

from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

from verdictai.ingestion.chunker.docling.options import (
    DOCLING_HIERARCHICAL_CHUNKER_NAME,
)
from verdictai.ingestion.chunker.docling.base_docling_chunker import BaseDoclingChunker
from verdictai.ingestion.chunker.docling.runtime.interfaces import (
    DoclingChunkerProtocol,
)
from verdictai.ingestion.parser.types import JsonValue


class DoclingHierarchicalChunker(BaseDoclingChunker):
    """Use Docling's `HierarchicalChunker` on serialized parser output."""

    chunker_name = DOCLING_HIERARCHICAL_CHUNKER_NAME

    def _create_docling_chunker(self) -> DoclingChunkerProtocol:
        """Create a configured Docling `HierarchicalChunker`."""

        return HierarchicalChunker(merge_list_items=self.config.merge_list_items)

    def _build_result_metadata(self) -> dict[str, JsonValue]:
        """Return metadata specific to hierarchical chunking."""

        return {"merge_list_items": self.config.merge_list_items}
