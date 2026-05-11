"""Schema for parser progress events."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..types import JsonValue, ProgressStage
from .document_parse_result import DocumentParseResult


class DocumentParseProgress(BaseModel):
    """Represent one parser progress event.

    Attributes:
        stage: Stable lifecycle stage identifier for the parse operation.
        message: Human-readable summary of the current stage.
        percent: Approximate completion percentage in the inclusive range 0..100.
        metadata: Optional stage-specific payload for observability or UI updates.
        report: Final `DocumentParseResult`, populated only on the `completed`
            event.
    """

    model_config = ConfigDict(frozen=True)

    stage: ProgressStage = Field(description="Stable parser stage identifier.")
    message: str = Field(description="Human-readable description of the current stage.")
    percent: int = Field(ge=0, le=100, description="Approximate stage completion.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional stage-specific metadata for observability.",
    )
    report: DocumentParseResult | None = Field(
        default=None,
        description="Final parse report attached to the completed event.",
    )
