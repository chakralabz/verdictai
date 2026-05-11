"""Docling-backed parser implementation."""

from __future__ import annotations

from ..schemas import DocumentFigureInspection, DocumentParseProgress
from .docling_parser import DoclingParser, DoclingParserConfig

__all__ = [
    "DoclingParser",
    "DoclingParserConfig",
    "DocumentFigureInspection",
    "DocumentParseProgress",
]
