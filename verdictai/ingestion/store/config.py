"""Configuration models for ingestion stores."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

StoreBackend = Literal["qdrant"]
QdrantDistance = Literal["cosine", "dot", "euclid", "manhattan"]
SparseModifier = Literal["none", "idf"]


class QdrantStoreConfig(BaseModel):
    """Configure the Qdrant-backed hybrid retrieval store.

    Attributes:
        url: Qdrant HTTP endpoint. The local compose file exposes this on
            `http://localhost:6333`.
        api_key: Optional Qdrant API key for secured deployments.
        collection_name: Collection used for embedded chunks.
        dense_vector_name: Named vector slot for dense embeddings.
        sparse_vector_name: Named vector slot for BM25 sparse vectors.
        sparse_model_name: FastEmbed sparse model used by Qdrant's Python
            client to infer BM25 vectors from chunk text.
        distance: Distance function for dense vector search.
        prefer_grpc: Whether the client should prefer gRPC calls.
        timeout_seconds: Request timeout passed to the Qdrant client.
        batch_size: Maximum points sent in one upsert request.
        dense_on_disk: Whether dense vectors should be stored on disk.
        sparse_on_disk: Whether sparse vector index data should be stored on
            disk.
        sparse_modifier: Optional Qdrant sparse-query modifier.
    """

    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "verdictai_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    sparse_model_name: str = "Qdrant/bm25"
    distance: QdrantDistance = "cosine"
    prefer_grpc: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0)
    batch_size: PositiveInt = 64
    dense_on_disk: bool = False
    sparse_on_disk: bool = False
    sparse_modifier: SparseModifier = "idf"


class StoreSettings(BaseModel):
    """Store package configuration namespace.

    Attributes:
        backend: Storage backend selected for ingestion writes.
        qdrant: Qdrant-specific store settings.
    """

    backend: StoreBackend = "qdrant"
    qdrant: QdrantStoreConfig = Field(default_factory=QdrantStoreConfig)
