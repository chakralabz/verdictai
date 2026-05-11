# Docling Chunker Adapter

This package adapts Docling chunkers to VerdictAI's ingestion contracts.
Application code should call a `DocumentChunkerProtocol` implementation and work
with `DocumentParseResult` and `ChunkingResult`; Docling runtime objects stay in
the `runtime/` subpackage.

## Boundary

- `hybrid.py`, `hierarchical.py`, and `line_based.py` are the public chunker
  implementations kept at package root.
- `options/` stores user-facing chunker configuration and stable backend names.
- `runtime/` restores serialized Docling documents, builds optional tokenizers
  and serializers, runs the selected Docling chunker, and normalizes output.

## Chunk Ownership

`Chunk` is the pre-embedding retrieval unit: text, contextualized text,
provenance, token counts, and chunker metadata. Embedding vectors are intentionally
not fields on `Chunk`; they belong to the embedding result or vector-store record
that knows the embedding model and vector dimensions.

## Implementations

- `hybrid.py`: heading-aware token chunking, the default production choice.
- `hierarchical.py`: structure-preserving chunking without token splitting.
- `line_based.py`: line-preserving token chunking for text-sensitive documents.

## Configuration

Use `DoclingChunkerConfig` from `options/` to configure tokenizer,
serializer, and Docling chunker options. Tokenizers should be aligned with the
downstream embedding model so `token_count` reflects the eventual embedding
budget.
