"""Schema for the full parsing outcome."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..types import JsonValue
from .parsed_block import ParsedBlock


class DocumentParseResult(BaseModel):
    """Structured result for one parsing request.

    Attributes:
        blocks: Parsed blocks emitted by the successful backend.
        metadata: Parser/orchestrator metadata useful for debugging and telemetry.
    """

    model_config = ConfigDict(frozen=True)

    blocks: list[ParsedBlock] = Field(
        description="Parsed blocks emitted by the successful backend."
    )

    metadata: dict[str, JsonValue] = Field(
        default_factory=dict, description="Additional metadata for parse result."
    )
