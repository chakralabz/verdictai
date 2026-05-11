"""Schema for figure-level parser inspection results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..types import ProvenanceEntry


class DocumentFigureInspection(BaseModel):
    """Represent one figure-level inspection record from a parsed document.

    Attributes:
        page: One-based page number when provenance is available.
        docling_label: Original Docling label associated with the figure item.
        has_image: Whether Docling exposed a renderable image for the figure.
        figure_ocr_text: Text extracted from the rendered figure image.
        figure_ocr_scores: Confidence scores returned by the secondary OCR pass.
        docling_is_ocr: Whether Docling marked the source content as OCR-derived.
        docling_confidence: Aggregate confidence extracted from Docling provenance.
        provenance: Normalized provenance payload from Docling.
        bbox: Bounding box in page space when available.
        tree_level: Original Docling tree depth for the figure item.
    """

    model_config = ConfigDict(frozen=True)

    page: int | None = Field(default=None, ge=1, description="One-based source page.")
    docling_label: str | None = Field(
        default=None, description="Original Docling figure label."
    )
    has_image: bool = Field(description="Whether the figure exposed an image payload.")
    figure_ocr_text: str = Field(
        default="", description="Text extracted from the figure image."
    )
    figure_ocr_scores: list[float] = Field(
        default_factory=list,
        description="Secondary OCR confidence scores for the figure image.",
    )
    docling_is_ocr: bool = Field(
        default=False, description="Whether Docling considered the source OCR-derived."
    )
    docling_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence from Docling provenance.",
    )
    provenance: list[ProvenanceEntry] = Field(
        default_factory=list,
        description="Normalized provenance payload emitted by Docling.",
    )
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Figure bounding box in page coordinate space when available.",
    )
    tree_level: int = Field(ge=0, description="Original Docling tree depth.")
