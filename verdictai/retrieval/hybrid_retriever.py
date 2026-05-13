"""Hybrid dense and sparse retrieval implementation."""

from __future__ import annotations

from time import perf_counter
from typing import Any, cast

from verdictai.config.settings import get_settings
from verdictai.ingestion.chunker.schemas import Chunk
from verdictai.ingestion.embeddings import (
    EmbedderProtocol,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from verdictai.ingestion.parser.types import JsonValue
from verdictai.ingestion.store.qdrant_store import QdrantStore
from verdictai.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Retrieve chunks using Qdrant native dense and sparse hybrid search.

    Uses the existing VerdictAI embedding provider for dense query vectors and
    FastEmbed BM25 sparse vectors for lexical recall. Qdrant's reciprocal rank
    fusion combines both candidate sets before the retriever converts scored
    points back into canonical chunk objects.

    Attributes:
        qdrant_store: Store instance that exposes hybrid vector search.
        sparse_model_name: FastEmbed model identifier used for BM25 queries.
        sparse_model: FastEmbed sparse text embedding model for BM25 queries.
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        sparse_model_name: str = "Qdrant/bm25",
    ) -> None:
        """Initialize hybrid retrieval dependencies.

        Args:
            qdrant_store: Qdrant-backed store used for hybrid vector search.
            sparse_model_name: FastEmbed sparse model name for BM25 query
                vectors. Defaults to `Qdrant/bm25`.

        Raises:
            ImportError: If FastEmbed is unavailable.
        """

        started_at = perf_counter()
        sparse_text_embedding_cls = _import_sparse_text_embedding()
        self.qdrant_store = qdrant_store
        self.sparse_model_name = sparse_model_name
        self.sparse_model = sparse_text_embedding_cls(model_name=sparse_model_name)
        self._dense_embedder: EmbedderProtocol | None = None
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "Loaded sparse retrieval model %s in %.2f ms.",
            sparse_model_name,
            elapsed_ms,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Retrieve top-k chunks with Qdrant dense and sparse RRF search.

        Args:
            query: Natural language legal question.
            top_k: Maximum chunks to return. Defaults to 5.
            filters: Optional Qdrant payload filter.

        Returns:
            Canonical chunks ranked by Qdrant reciprocal rank fusion.

        Raises:
            RuntimeError: If Qdrant hybrid search fails.
        """

        if top_k <= 0:
            return []

        started_at = perf_counter()
        dense_vec = self._get_dense_embedder().generate_embedding(query)
        sparse_indices, sparse_values = _compute_sparse_query(
            sparse_model=self.sparse_model,
            query=query,
        )
        points = self.qdrant_store.hybrid_search(
            dense_vec=dense_vec,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=top_k * 4,
            filters=filters,
        )
        chunks = [_point_to_chunk(point) for point in points[:top_k]]
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "Retrieved %d chunks for query %r in %.2f ms.",
            len(chunks),
            query[:50],
            elapsed_ms,
        )
        return chunks

    def _get_dense_embedder(self) -> EmbedderProtocol:
        """Return the configured dense embedding provider."""

        if self._dense_embedder is not None:
            return self._dense_embedder

        config = get_settings().embeddings
        if config.backend == "openai":
            self._dense_embedder = OpenAIEmbeddingProvider(config)
        else:
            self._dense_embedder = SentenceTransformerEmbeddingProvider(config)
        return self._dense_embedder


def _import_sparse_text_embedding() -> Any:
    """Import FastEmbed's sparse text embedding model lazily.

    Returns:
        FastEmbed `SparseTextEmbedding` class.

    Raises:
        ImportError: If FastEmbed is unavailable.
    """

    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise ImportError(
            "Install `fastembed` to compute BM25 sparse query vectors."
        ) from exc
    return SparseTextEmbedding


def _compute_sparse_query(*, sparse_model: Any, query: str) -> tuple[list[int], list[float]]:
    """Compute one BM25 sparse query vector.

    Args:
        sparse_model: FastEmbed sparse text embedding model.
        query: Query text to encode.

    Returns:
        Tuple containing sparse vector indices and values.
    """

    sparse_vector = next(iter(sparse_model.embed([query])))
    return (
        [int(value) for value in _array_to_list(sparse_vector.indices)],
        [float(value) for value in _array_to_list(sparse_vector.values)],
    )


def _point_to_chunk(point: Any) -> Chunk:
    """Convert a Qdrant scored point payload into a canonical chunk."""

    payload = getattr(point, "payload", None)
    if not isinstance(payload, dict):
        raise RuntimeError("Qdrant search result is missing chunk payload.")

    metadata = _metadata_dict(payload.get("metadata"))
    text = str(payload.get("text") or "")
    chunk_metadata = _metadata_dict(metadata.get("chunk_metadata")) or metadata
    return Chunk(
        doc_id=_optional_str(payload.get("doc_id")),
        source_path=_optional_str(payload.get("source_path")),
        source_name=_optional_str(payload.get("source_name")),
        chunk_id=str(payload["chunk_id"]),
        chunk_index=int(payload["chunk_index"]),
        chunker_used=str(metadata.get("chunker_used") or "unknown"),
        text=text,
        contextualized_text=text,
        headings=_string_list(metadata.get("headings")),
        captions=_string_list(metadata.get("captions")),
        page_start=_optional_int(metadata.get("page_start")),
        page_end=_optional_int(metadata.get("page_end")),
        bbox=_bbox(metadata.get("bbox")),
        token_count=_optional_int(metadata.get("token_count")),
        char_count=int(metadata.get("char_count") or len(text)),
        metadata=chunk_metadata,
    )


def _metadata_dict(value: Any) -> dict[str, JsonValue]:
    """Return a JSON-safe metadata mapping when available."""

    if not isinstance(value, dict):
        return {}
    return cast(dict[str, JsonValue], value)


def _optional_str(value: Any) -> str | None:
    """Return a string value or None."""

    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    """Return an integer value or None."""

    if value is None:
        return None
    return int(value)


def _string_list(value: Any) -> list[str]:
    """Return a list of string values."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Return a normalized bounding box tuple when available."""

    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    return tuple(float(item) for item in value)


def _array_to_list(values: Any) -> list[Any]:
    """Convert FastEmbed numpy-like values into plain Python lists."""

    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)
