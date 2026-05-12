"""Parser interfaces, schemas, and Docling-backed implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .docling import DoclingParserConfig
from .document_parser import DocumentParserProtocol
from .schemas import (
    DocumentFigureInspection,
    DocumentParseProgress,
    DocumentParseResult,
    ParsedBlock,
)

if TYPE_CHECKING:
    from .docling import DoclingParser

__all__ = [
    "DoclingParser",
    "DoclingParserConfig",
    "DocumentFigureInspection",
    "DocumentParseProgress",
    "DocumentParseResult",
    "DocumentParserProtocol",
    "ParsedBlock",
]


def __getattr__(name: str) -> object:
    """Import concrete Docling parser only when requested."""

    if name == "DoclingParser":
        from .docling import DoclingParser

        return DoclingParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
