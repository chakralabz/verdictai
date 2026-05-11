"""Shared parser-facing type contracts.

This module centralizes small reusable contracts so parser implementations and
Pydantic schemas can share the same typed shapes instead of repeating
`dict[str, Any]` throughout the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonPrimitiveList: TypeAlias = list[JsonPrimitive]
JsonObjectValue: TypeAlias = JsonPrimitive | JsonPrimitiveList
JsonObject: TypeAlias = dict[str, JsonObjectValue]
JsonValue: TypeAlias = JsonObjectValue | JsonObject | list[JsonObject]

ProgressStage = Literal[
    "validating_input",
    "configuring_environment",
    "importing_backend",
    "building_converter",
    "converting",
    "conversion_complete",
    "normalizing_blocks",
    "building_metadata",
    "completed",
]

class ProvenanceEntry(TypedDict):
    """Normalized Docling provenance payload for one source fragment."""

    page_no: int | None
    bbox: list[float] | None
    charspan: list[int] | None
    confidence: float | None
    source: str | None


class PipelineMetadata(TypedDict):
    """Serializable summary of the configured Docling pipeline."""

    pipeline: str
    ocr_enabled: bool
    table_structure_enabled: bool
    ocr_engine: str | None
    vlm_model: str | None
    force_backend_text: bool
    picture_images_enabled: bool
    remote_services_enabled: bool
    model_cache_dir: str | None
    artifacts_dir: str | None


@dataclass(slots=True, frozen=True)
class ExtractedProvenance:
    """Structured provenance extracted from a Docling item."""

    page_number: int | None
    bbox: tuple[float, float, float, float] | None
    average_confidence: float | None
    is_ocr: bool
    entries: list[ProvenanceEntry]
