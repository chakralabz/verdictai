"""Runtime contract for ingestion-facing document parsers.

The ingestion pipeline depends on behavior, not on the concrete orchestrator
class. Keeping this protocol small makes it easy to substitute test doubles or
future parser orchestrators without widening the integration surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from verdictai.ingestion.parser.schemas import (
    DocumentParseProgress,
    DocumentParseResult,
    ParsedBlock,
)


@runtime_checkable
class DocumentParserProtocol(Protocol):
    """Define the parser capabilities required by the ingestion pipeline."""

    def parse_document(self, path: str | Path) -> list[ParsedBlock]:
        """Parse a document into canonical blocks.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            List of `ParsedBlock` objects in reading order.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file or is otherwise invalid.
            RuntimeError: For backend-specific parsing failures.
        """

    def parse_with_report(self, path: str | Path) -> DocumentParseResult:
        """Parse a document and return blocks plus structured metadata.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            A `DocumentParseResult` containing the blocks and parse metadata.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file or is otherwise invalid.
            RuntimeError: For backend-specific parsing failures.
        """

    def parse_with_progress(
        self,
        path: str | Path,
    ) -> AsyncIterator[DocumentParseProgress]:
        """Yield structured progress events for a parsing request.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Yields:
            `DocumentParseProgress` events in deterministic stage order.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file or is otherwise invalid.
            RuntimeError: For backend-specific parsing failures.

        Notes:
            Implementations should emit a terminal `completed` event whose `report`
            field contains the final `DocumentParseResult`.
        """
