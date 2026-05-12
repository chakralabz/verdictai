"""Schema for a complete embedding request."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from verdictai.ingestion.embeddings.schemas.embedded_chunk import EmbeddedChunk
from verdictai.ingestion.parser.types import JsonValue


class EmbeddingResult(BaseModel):
    """Structured result for one chunk embedding request.

    Attributes:
        chunks: Embedded chunks in the same order as the input chunks.
        metadata: Request-level provider metadata for debugging and telemetry.
    """

    model_config = ConfigDict(frozen=True)

    chunks: list[EmbeddedChunk] = Field(description="Embedded chunks.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional metadata for the embedding request.",
    )
