"""Integration-style tests for Qdrant hybrid store behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from verdictai.ingestion.embeddings.schemas import EmbeddedChunk
from verdictai.ingestion.store.config import QdrantStoreConfig
from verdictai.ingestion.store.qdrant_store import QdrantStore

pytestmark = pytest.mark.integration


class FakeSparseTextEmbedding:
    """Fake FastEmbed sparse model for deterministic BM25 vectors."""

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[SimpleNamespace]:
        """Return stable sparse vectors for each text string."""

        return [
            SimpleNamespace(
                indices=[index + 1, index + 10],
                values=[float(len(text)), float(index + 1)],
            )
            for index, text in enumerate(texts)
        ]


class FakeVectorParams:
    """Capture dense vector collection configuration."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeSparseIndexParams:
    """Capture sparse index collection configuration."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeSparseVectorParams:
    """Capture sparse vector collection configuration."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeSparseVector:
    """Capture a Qdrant sparse vector."""

    def __init__(self, *, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class FakePointStruct:
    """Capture a Qdrant point payload."""

    def __init__(
        self,
        *,
        id: str,
        vector: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class FakePrefetch:
    """Capture a Qdrant prefetch query."""

    def __init__(self, *, query: dict[str, Any], limit: int) -> None:
        self.query = query
        self.limit = limit


class FakeFusionQuery:
    """Capture a Qdrant fusion query."""

    def __init__(self, *, fusion: str) -> None:
        self.fusion = fusion


class FakeModels:
    """Namespace for Qdrant model fakes used by the store."""

    class Distance:
        """Fake dense vector distance constants."""

        COSINE = "cosine"
        DOT = "dot"
        EUCLID = "euclid"
        MANHATTAN = "manhattan"

    class Modifier:
        """Fake sparse modifier constants."""

        IDF = "idf"

    class Fusion:
        """Fake Qdrant fusion constants."""

        RRF = "rrf"

    VectorParams = FakeVectorParams
    SparseIndexParams = FakeSparseIndexParams
    SparseVectorParams = FakeSparseVectorParams
    SparseVector = FakeSparseVector
    PointStruct = FakePointStruct
    Prefetch = FakePrefetch
    FusionQuery = FakeFusionQuery


class FakeQdrantClient:
    """In-memory fake for Qdrant collection writes and hybrid queries."""

    collections: ClassVar[dict[str, dict[str, FakePointStruct]]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.last_query: dict[str, Any] | None = None

    def collection_exists(self, *, collection_name: str) -> bool:
        """Return whether the fake collection has been created."""

        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, **kwargs: Any) -> None:
        """Create an empty fake collection."""

        self.collections[collection_name] = {}
        self.collection_kwargs = kwargs

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[FakePointStruct],
        wait: bool,
    ) -> None:
        """Store points by ID to mirror idempotent Qdrant upsert semantics."""

        collection = self.collections.setdefault(collection_name, {})
        for point in points:
            collection[point.id] = point
        self.wait = wait

    def query_points(self, *, collection_name: str, **kwargs: Any) -> SimpleNamespace:
        """Return fake scored points from the requested collection."""

        self.last_query = kwargs
        points = list(self.collections[collection_name].values())
        limit = kwargs["limit"]
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=point.id, payload=point.payload, score=1.0)
                for point in points[:limit]
            ]
        )

    def close(self) -> None:
        """Close the fake client."""


@pytest.fixture(autouse=True)
def fake_qdrant_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Qdrant and FastEmbed dependencies with deterministic fakes."""

    from verdictai.ingestion.store import qdrant_store

    FakeQdrantClient.collections = {}
    monkeypatch.setattr(
        qdrant_store,
        "_import_qdrant",
        lambda: (object, FakeQdrantClient, FakeModels),
    )
    monkeypatch.setattr(
        qdrant_store,
        "_import_sparse_text_embedding",
        lambda: FakeSparseTextEmbedding,
    )


def test_store_embeddings_upserts_sparse_vectors_and_hybrid_search_returns_them() -> None:
    """Upsert chunks with dense and sparse vectors, then retrieve via RRF search."""

    store = QdrantStore(config=QdrantStoreConfig(collection_name="test_chunks"))
    chunks = _embedded_chunks(count=3)

    result = store.store_embeddings(chunks)
    hits = store.hybrid_search(
        dense_vec=[0.1, 0.2, 0.3],
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.25],
        top_k=3,
        filters={"must": []},
    )

    stored_points = FakeQdrantClient.collections["test_chunks"].values()
    assert result.stored_count == 3
    assert len(hits) == 3
    assert all(
        isinstance(point.vector["bm25"], FakeSparseVector) for point in stored_points
    )
    assert store._client.last_query["query"].fusion == "rrf"
    assert store._client.last_query["query_filter"] == {"must": []}


def test_store_embeddings_is_idempotent_for_same_chunk_id() -> None:
    """Upserting the same chunk twice keeps one point in the collection."""

    store = QdrantStore(config=QdrantStoreConfig(collection_name="test_chunks"))
    chunk = _embedded_chunks(count=1)[0]

    store.store_embeddings([chunk])
    store.store_embeddings([chunk])

    assert len(FakeQdrantClient.collections["test_chunks"]) == 1


def _embedded_chunks(*, count: int) -> list[EmbeddedChunk]:
    """Build embedded chunks with stable IDs and dimensions."""

    return [
        EmbeddedChunk(
            chunk_id=f"chunk-{index}",
            chunk_index=index,
            doc_id="doc-001",
            source_path="/tmp/example-contract.pdf",
            source_name="example-contract.pdf",
            text=f"Clause {index} discusses liability and Delaware law.",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model",
            embedding_provider="test-provider",
            metadata={"page_start": index + 1},
        )
        for index in range(count)
    ]
