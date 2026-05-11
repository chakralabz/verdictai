"""Normalization helpers for Docling chunk outputs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

from verdictai.ingestion.chunker.docling.runtime.interfaces import (
    ChunkContextualizer,
    DoclingChunkerProtocol,
    DoclingChunkProtocol,
    TokenCounter,
)
from verdictai.ingestion.chunker.schemas import Chunk
from verdictai.ingestion.parser.schemas import DocumentParseResult
from verdictai.ingestion.parser.types import JsonValue


def normalize_docling_chunks(
    *,
    chunker_name: str,
    docling_chunker: DoclingChunkerProtocol,
    docling_chunks: list[DoclingChunkProtocol],
    parse_result: DocumentParseResult,
) -> list[Chunk]:
    """Normalize Docling chunks into VerdictAI chunk records.

    Args:
        chunker_name: Stable name of the Docling chunker implementation.
        docling_chunker: Configured Docling chunker that emitted the chunks.
        docling_chunks: Docling chunks in deterministic emission order.
        parse_result: Parser report that carries source document metadata.

    Returns:
        Canonical `Chunk` records suitable for embedding and retrieval.

    Notes:
        `Chunk` intentionally does not include an embedding vector. Embeddings
        are model-specific artifacts produced after chunking and should be stored
        in a downstream embedding result or vector-store record.
    """

    document_id = _coerce_string(parse_result.metadata.get("document_id"))
    source_path = _coerce_string(parse_result.metadata.get("source_path"))
    source_name = _coerce_string(parse_result.metadata.get("source_name"))

    normalized_chunks: list[Chunk] = []
    for chunk_index, docling_chunk in enumerate(docling_chunks):
        text = _extract_chunk_text(docling_chunk)
        contextualized_text = _contextualize_chunk(
            docling_chunker=docling_chunker,
            docling_chunk=docling_chunk,
            fallback_text=text,
        )
        headings, captions = _extract_heading_context(docling_chunk)
        page_start, page_end = _extract_page_window(docling_chunk)
        bbox = _extract_first_bbox(docling_chunk)
        token_count = _count_tokens(
            docling_chunker=docling_chunker, text=contextualized_text
        )

        normalized_chunks.append(
            Chunk(
                chunk_id=_build_chunk_id(
                    source_path=source_path,
                    chunker_name=chunker_name,
                    chunk_index=chunk_index,
                    text=contextualized_text,
                ),
                doc_id=document_id,
                source_path=source_path,
                source_name=source_name,
                chunker_used=chunker_name,
                chunk_index=chunk_index,
                text=text,
                contextualized_text=contextualized_text,
                headings=headings,
                captions=captions,
                page_start=page_start,
                page_end=page_end,
                bbox=bbox,
                token_count=token_count,
                char_count=len(contextualized_text),
                metadata=_build_chunk_metadata(docling_chunk),
            )
        )

    return normalized_chunks


def _contextualize_chunk(
    *,
    docling_chunker: DoclingChunkerProtocol,
    docling_chunk: DoclingChunkProtocol,
    fallback_text: str,
) -> str:
    """Return metadata-enriched chunk text when supported by the chunker."""

    if isinstance(docling_chunker, ChunkContextualizer):
        try:
            contextualized = docling_chunker.contextualize(docling_chunk)
        except Exception:  # pragma: no cover - depends on docling runtime
            contextualized = None
        if isinstance(contextualized, str) and contextualized.strip():
            return contextualized
    return fallback_text


def _extract_heading_context(
    docling_chunk: DoclingChunkProtocol,
) -> tuple[list[str], list[str]]:
    """Extract headings and captions from a Docling chunk."""

    meta = _read_docling_attribute(docling_chunk, "meta")
    headings = _coerce_string_list(_read_docling_attribute(meta, "headings"))
    captions = _coerce_string_list(_read_docling_attribute(meta, "captions"))
    return headings, captions


def _extract_page_window(
    docling_chunk: DoclingChunkProtocol,
) -> tuple[int | None, int | None]:
    """Extract the source page range covered by a Docling chunk."""

    meta = _read_docling_attribute(docling_chunk, "meta")
    origin = _read_docling_attribute(meta, "origin")
    page_candidates = _collect_page_candidates(origin)

    # 1. Prefer the explicit origin metadata when Docling exposes it.
    if page_candidates:
        return min(page_candidates), max(page_candidates)

    # 2. Fall back to provenance on the underlying doc items for chunk types that
    #    do not populate a direct `origin` page window.
    doc_items = _as_sequence(_read_docling_attribute(meta, "doc_items"))
    if doc_items is not None:
        for item in doc_items:
            page_candidates.extend(_collect_page_candidates_from_item(item))

    if not page_candidates:
        return None, None
    return min(page_candidates), max(page_candidates)


def _collect_page_candidates(origin: object | None) -> list[int]:
    """Collect page numbers from a chunk origin payload."""

    if origin is None:
        return []

    page_candidates: list[int] = []
    for attribute in ("page_no", "page", "page_start", "page_end"):
        value = _read_docling_attribute(origin, attribute)
        if isinstance(value, int) and value >= 1:
            page_candidates.append(value)

    pages = _as_sequence(_read_docling_attribute(origin, "pages"))
    if pages is not None:
        page_candidates.extend(
            page for page in pages if isinstance(page, int) and page >= 1
        )

    return page_candidates


def _collect_page_candidates_from_item(item: object) -> list[int]:
    """Collect page numbers from a doc item's provenance records."""

    page_candidates: list[int] = []
    provenance_records = _as_sequence(_read_docling_attribute(item, "prov"))
    if provenance_records is None:
        return page_candidates

    for prov in provenance_records:
        page_no = _read_docling_attribute(prov, "page_no")
        if isinstance(page_no, int) and page_no >= 1:
            page_candidates.append(page_no)
    return page_candidates


def _extract_first_bbox(
    docling_chunk: DoclingChunkProtocol,
) -> tuple[float, float, float, float] | None:
    """Return the first bounding box covered by a chunk when available."""

    bboxes = _collect_bboxes(docling_chunk)
    if not bboxes:
        return None
    return tuple(bboxes[0])


def _build_chunk_metadata(docling_chunk: DoclingChunkProtocol) -> dict[str, JsonValue]:
    """Build a JSON-safe metadata dictionary for one Docling chunk."""

    metadata: dict[str, JsonValue] = {"docling_chunk_type": type(docling_chunk).__name__}
    meta = _read_docling_attribute(docling_chunk, "meta")

    for field_name in ("schema_name", "section", "label"):
        value = _read_docling_attribute(meta, field_name)
        if isinstance(value, str | int | float | bool) or value is None:
            metadata[field_name] = value
        elif (sequence := _as_sequence(value)) is not None:
            metadata[field_name] = [str(item) for item in sequence]

    doc_items = _as_sequence(_read_docling_attribute(meta, "doc_items"))
    if doc_items is not None:
        doc_item_refs: list[object] = []
        for item in doc_items:
            ref = _read_docling_attribute(item, "self_ref")
            if isinstance(ref, str) and ref:
                doc_item_refs.append(ref)
        metadata["doc_item_refs"] = doc_item_refs

    bboxes = _collect_bboxes(docling_chunk)
    if bboxes:
        metadata["bboxes"] = [list(bbox) for bbox in bboxes]

    return metadata


def _collect_bboxes(
    docling_chunk: DoclingChunkProtocol,
) -> list[tuple[float, float, float, float]]:
    """Collect unique provenance boxes for all source items in a chunk.

    Args:
        docling_chunk: Chunk emitted by Docling.

    Returns:
        Bounding boxes in Docling page coordinate space, ordered by first
        occurrence in the chunk provenance.
    """

    meta = _read_docling_attribute(docling_chunk, "meta")
    bboxes: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()

    origin = _read_docling_attribute(meta, "origin")
    origin_bbox = _normalize_bbox(_read_docling_attribute(origin, "bbox"))
    if origin_bbox is not None:
        seen.add(origin_bbox)
        bboxes.append(origin_bbox)

    doc_items = _as_sequence(_read_docling_attribute(meta, "doc_items"))
    if doc_items is None:
        return bboxes

    for item in doc_items:
        item_bbox = _normalize_bbox(_read_docling_attribute(item, "bbox"))
        if item_bbox is not None and item_bbox not in seen:
            seen.add(item_bbox)
            bboxes.append(item_bbox)

        provenance_records = _as_sequence(_read_docling_attribute(item, "prov"))
        if provenance_records is None:
            continue
        for prov in provenance_records:
            bbox = _normalize_bbox(_read_docling_attribute(prov, "bbox"))
            if bbox is not None and bbox not in seen:
                seen.add(bbox)
                bboxes.append(bbox)
    return bboxes


def _normalize_bbox(bbox: object | None) -> tuple[float, float, float, float] | None:
    """Normalize Docling bbox shapes into `(left, top, right, bottom)`."""

    if bbox is None:
        return None

    if isinstance(bbox, Sequence) and not isinstance(bbox, str | bytes):
        if len(bbox) != 4:
            return None
        values = [_coerce_float(value) for value in bbox]
        if all(value is not None for value in values):
            return cast(tuple[float, float, float, float], tuple(values))

    values = []
    for attribute in ("l", "t", "r", "b"):
        value = _coerce_float(_read_docling_attribute(bbox, attribute))
        if value is None:
            values = []
            break
        values.append(value)
    if len(values) == 4:
        return tuple(values)

    values = []
    for attribute in ("left", "top", "right", "bottom"):
        value = _coerce_float(_read_docling_attribute(bbox, attribute))
        if value is None:
            return None
        values.append(value)
    return tuple(values) if len(values) == 4 else None


def _count_tokens(*, docling_chunker: DoclingChunkerProtocol, text: str) -> int | None:
    """Count tokens for a chunk when the Docling tokenizer exposes the method."""

    tokenizer = _read_docling_attribute(docling_chunker, "tokenizer")
    if not isinstance(tokenizer, TokenCounter):
        return None

    try:
        result = tokenizer.count_tokens(text)
    except Exception:  # pragma: no cover - depends on docling runtime
        return None
    return result if result >= 0 else None


def _extract_chunk_text(docling_chunk: DoclingChunkProtocol) -> str:
    """Extract the base text representation from a Docling chunk."""

    text = _read_docling_attribute(docling_chunk, "text")
    if isinstance(text, str):
        return text
    return str(docling_chunk)


def _build_chunk_id(
    *,
    source_path: str | None,
    chunker_name: str,
    chunk_index: int,
    text: str,
) -> str:
    """Build a deterministic chunk identifier."""

    fingerprint = "|".join(
        [
            source_path or "",
            chunker_name,
            str(chunk_index),
            " ".join(text.split())[:200],
        ]
    )
    return hashlib.sha1(
        fingerprint.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def _coerce_string(value: JsonValue | None) -> str | None:
    """Normalize an optional metadata string."""

    return value if isinstance(value, str) else None


def _coerce_string_list(value: object | None) -> list[str]:
    """Normalize an optional iterable of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if (sequence := _as_sequence(value)) is not None:
        return [str(item) for item in sequence if str(item)]
    return [str(value)]


def _coerce_float(value: object | None) -> float | None:
    """Return a finite float for numeric Docling provenance values."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _read_docling_attribute(source: object | None, name: str) -> object | None:
    """Read an optional attribute from a Docling object at the adapter boundary.

    Docling exposes several Pydantic/dataclass-like runtime objects that are not
    part of VerdictAI's public type contract. Keeping optional attribute access
    in this helper prevents reflective code from leaking into application logic.
    """

    if source is None:
        return None
    try:
        return object.__getattribute__(source, name)
    except AttributeError:
        return None


def _as_sequence(value: object | None) -> Sequence[object] | None:
    """Return `value` as a non-string sequence when possible."""

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return None
