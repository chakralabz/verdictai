"""Schemas for embedded retrieval chunks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from verdictai.ingestion.parser.types import JsonValue


class EmbeddedChunk(BaseModel):
    """Chunk plus the model-specific vector generated for retrieval.

    Attributes:
        chunk_id: Source chunk identifier.
        chunk_index: Source chunk index in emission order.
        doc_id: Source document identifier when available.
        source_path: Source document path when available.
        source_name: Source document filename when available.
        text: Exact text sent to the embedding model.
        embedding: Embedding vector returned by the provider.
        embedding_model: Model identifier used for this vector.
        embedding_provider: Provider backend used for this vector.
        embedding_dimension: Number of floats in `embedding`.
        metadata: JSON-safe provenance and provider metadata.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    chunk_id: str = Field(description="Source chunk identifier.")
    chunk_index: int = Field(ge=0, description="Source chunk emission index.")
    doc_id: str | None = Field(default=None, description="Source document ID.")
    source_path: str | None = Field(default=None, description="Filesystem source path.")
    source_name: str | None = Field(default=None, description="Source filename.")
    text: str = Field(description="Text sent to the embedding model.")
    embedding: list[float] = Field(description="Embedding vector.")
    embedding_model: str = Field(description="Embedding model identifier.")
    embedding_provider: str = Field(description="Embedding provider backend.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional chunk and provider metadata.",
    )

    @property
    def embedding_dimension(self) -> int:
        """Return the vector length."""

        return len(self.embedding)
