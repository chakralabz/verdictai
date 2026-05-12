"""Docling-backed parser implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas import DocumentFigureInspection, DocumentParseProgress
from .docling_parser_config import DoclingParserConfig

if TYPE_CHECKING:
    from .docling_parser import DoclingParser

__all__ = [
    "DoclingParser",
    "DoclingParserConfig",
    "DocumentFigureInspection",
    "DocumentParseProgress",
]


def __getattr__(name: str) -> object:
    """Import the concrete parser only when callers request it."""

    if name == "DoclingParser":
        from .docling_parser import DoclingParser

        return DoclingParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
