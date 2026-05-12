"""Schema for store write results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from verdictai.ingestion.parser.types import JsonValue


class StoreResult(BaseModel):
    """Structured result for one store write request.

    Attributes:
        backend: Store implementation that handled the write.
        collection_name: Backend collection or index name.
        point_ids: Backend point identifiers in input chunk order.
        stored_count: Number of chunks accepted for storage.
        metadata: Additional backend-neutral write metadata.
    """

    model_config = ConfigDict(frozen=True)

    backend: str = Field(description="Store backend that handled the write.")
    collection_name: str = Field(description="Collection or index written to.")
    point_ids: list[str] = Field(description="Stored point IDs in input order.")
    stored_count: int = Field(ge=0, description="Number of stored chunks.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional write metadata.",
    )
