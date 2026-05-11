"""Parser interfaces, schemas, and Docling-backed implementations."""

from __future__ import annotations

from .api import (
    collect_document_progress,
    create_parser,
    parse_document,
    parse_document_async,
)
from .docling import DoclingParser, DoclingParserConfig
from .document_parser import DocumentParserProtocol
from .schemas import (
    DocumentFigureInspection,
    DocumentParseProgress,
    DocumentParseResult,
    ParsedBlock,
)

__all__ = [
    "DoclingParser",
    "DoclingParserConfig",
    "DocumentFigureInspection",
    "DocumentParseProgress",
    "DocumentParseResult",
    "DocumentParserProtocol",
    "ParsedBlock",
    "collect_document_progress",
    "create_parser",
    "parse_document",
    "parse_document_async",
]
