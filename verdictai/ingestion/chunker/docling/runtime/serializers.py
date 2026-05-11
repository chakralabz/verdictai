"""Serializer-provider helpers for advanced Docling chunking setups."""

from __future__ import annotations

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)

from verdictai.ingestion.chunker.docling.options import DoclingSerializerConfig


def build_serializer_provider(
    *,
    config: DoclingSerializerConfig,
) -> object | None:
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

    params_kwargs: dict[str, object] = {}
    if config.compact_tables:
        params_kwargs["compact_tables"] = True
    if config.image_placeholder is not None:
        params_kwargs["image_placeholder"] = config.image_placeholder

    markdown_params = MarkdownParams(**params_kwargs) if params_kwargs else None

    table_serializer = MarkdownTableSerializer() if use_markdown_tables else None

    def get_serializer(self: object, doc: object) -> object:
        """Apply configured serialization options to one document."""

        serializer_kwargs: dict[str, object] = {"doc": doc}
        if table_serializer is not None:
            serializer_kwargs["table_serializer"] = table_serializer
        if markdown_params is not None:
            serializer_kwargs["params"] = markdown_params
        return ChunkingDocSerializer(**serializer_kwargs)

    provider_cls = type(
        "VerdictAISerializerProvider",
        (ChunkingSerializerProvider,),
        {"get_serializer": get_serializer},
    )
    return provider_cls()
