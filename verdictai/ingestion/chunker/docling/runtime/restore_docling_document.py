"""Helpers for reconstructing Docling documents from parser output."""

from __future__ import annotations

from docling_core.types.doc.document import DoclingDocument

from verdictai.ingestion.chunker.docling.runtime.interfaces import (
    DoclingDocumentProtocol,
)
from verdictai.ingestion.parser.schemas import DocumentParseResult
from verdictai.utils.errors import (
    CHUNKER_DOCLING_METADATA_INVALID,
    CHUNKER_DOCLING_METADATA_MISSING,
    DoclingChunkerError,
)


def restore_docling_document(
    *,
    parse_result: DocumentParseResult,
) -> DoclingDocumentProtocol:
    """Restore a `DoclingDocument` from serialized parser metadata.

    Args:
        parse_result: Parser output containing a serialized `docling_document`.

    Returns:
        Reconstructed `DoclingDocument`.

    Raises:
        DoclingChunkerError: If the parse result does not carry a valid serialized
            Docling document payload.
    """

    payload = parse_result.metadata.get("docling_document")
    if not isinstance(payload, dict):
        raise DoclingChunkerError(CHUNKER_DOCLING_METADATA_MISSING)

    try:
        return DoclingDocument.model_validate(payload)
    except Exception as exc:
        raise DoclingChunkerError(CHUNKER_DOCLING_METADATA_INVALID) from exc
