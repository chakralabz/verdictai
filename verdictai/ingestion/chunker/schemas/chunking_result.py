"""Schema for the full chunking outcome."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from verdictai.ingestion.chunker.schemas.chunk import Chunk
from verdictai.ingestion.parser.types import JsonValue


class ChunkingResult(BaseModel):
    """Structured result for one chunking request.

    Attributes:
        chunks: Chunks emitted by the successful backend.
        metadata: Chunker metadata useful for debugging and telemetry.
    """

    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk] = Field(description="Chunks emitted by the successful backend.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional metadata for chunking result.",
    )
