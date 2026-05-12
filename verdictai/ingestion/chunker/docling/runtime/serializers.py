"""Serializer-provider helpers for advanced Docling chunking setups."""

from __future__ import annotations

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import DoclingDocument

from verdictai.ingestion.chunker.docling.options import DoclingSerializerConfig


class _VerdictAISerializerProvider(ChunkingSerializerProvider):
    """Serializer provider that applies VerdictAI's Docling serializer config."""

    def __init__(
        self,
        *,
        table_serializer: MarkdownTableSerializer | None,
        markdown_params: MarkdownParams | None,
    ) -> None:
        super().__init__()
        self._table_serializer = table_serializer
        self._markdown_params = markdown_params

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        """Apply configured serialization options to one document."""

        if self._table_serializer is not None and self._markdown_params is not None:
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=self._table_serializer,
                params=self._markdown_params,
            )
        if self._table_serializer is not None:
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=self._table_serializer,
            )
        if self._markdown_params is not None:
            return ChunkingDocSerializer(doc=doc, params=self._markdown_params)
        return ChunkingDocSerializer(doc=doc)


def build_serializer_provider(
    *,
    config: DoclingSerializerConfig,
) -> BaseSerializerProvider | None:
    """Build an optional Docling serializer provider from VerdictAI config.

    Args:
        config: Serializer configuration.

    Returns:
        A `ChunkingSerializerProvider` instance when custom serialization is
        requested, otherwise None.
    """

    use_markdown_tables = config.table_mode == "markdown"
    use_markdown_params = config.compact_tables or config.image_placeholder is not None
    if not use_markdown_tables and not use_markdown_params:
        return None

    markdown_params: MarkdownParams | None = None
    if config.compact_tables and config.image_placeholder is not None:
        markdown_params = MarkdownParams(
            compact_tables=True,
            image_placeholder=config.image_placeholder,
        )
    elif config.compact_tables:
        markdown_params = MarkdownParams(compact_tables=True)
    elif config.image_placeholder is not None:
        markdown_params = MarkdownParams(image_placeholder=config.image_placeholder)

    table_serializer = MarkdownTableSerializer() if use_markdown_tables else None
    return _VerdictAISerializerProvider(
        table_serializer=table_serializer,
        markdown_params=markdown_params,
    )
