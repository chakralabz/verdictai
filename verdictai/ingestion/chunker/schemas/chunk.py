"""Canonical chunk schema used by the chunking layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from verdictai.ingestion.parser.types import JsonValue


class Chunk(BaseModel):
    """Canonical retrieval chunk emitted by the chunking layer.

    Notes:
        `Chunk` represents the stable text and provenance unit before embedding.
        It deliberately does not carry an embedding vector because vectors are
        model-specific, dimension-specific artifacts owned by the embedding or
        vector-store layer.

    Attributes:
        doc_id: Stable identifier for the source document.
        source_path: Absolute filesystem path to the source document.
        source_name: Source filename.
        chunk_id: Deterministic identifier for this chunk.
        chunk_index: Stable 0-based index in emission order.
        chunker_used: Name of the chunker backend that produced this chunk.
        text: Backend chunk text emitted before contextualization.
        contextualized_text: Chunk text enriched with heading or caption context.
        headings: Heading lineage attached to the chunk.
        captions: Captions attached to the chunk.
        page_start: First source page covered by the chunk when available.
        page_end: Last source page covered by the chunk when available.
        bbox: First bounding box covered by the chunk, when available.
        token_count: Token count for `contextualized_text` when available.
        char_count: Character count for `contextualized_text`.
        metadata: Additional chunker-specific metadata.
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    doc_id: str | None = Field(default=None, description="Stable source document ID.")
    source_path: str | None = Field(default=None, description="Filesystem source path.")
    source_name: str | None = Field(default=None, description="Source filename.")
    chunk_id: str = Field(description="Deterministic chunk identifier.")
    chunk_index: int = Field(ge=0, description="Stable 0-based emission index.")
    chunker_used: str = Field(description="Chunker backend that produced the chunk.")
    text: str = Field(
        default="",
        description="Backend chunk text emitted before contextualization.",
    )
    contextualized_text: str = Field(
        default="",
        description="Chunk text enriched with additional context.",
    )
    headings: list[str] = Field(
        default_factory=list,
        description="Heading lineage attached to the chunk.",
    )
    captions: list[str] = Field(
        default_factory=list,
        description="Captions attached to the chunk.",
    )
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="First source page covered by the chunk when available.",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="Last source page covered by the chunk when available.",
    )
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="First source bounding box covered by the chunk when available.",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Token count for the contextualized text when available.",
    )
    char_count: int = Field(
        ge=0,
        description="Character count for the contextualized text.",
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional chunker metadata.",
    )
