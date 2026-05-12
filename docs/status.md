# VerdictAI Current Status

Last Updated: 2026-05-12

## Resume Here

Next recommended task: continue P1 reconciliation by checking `verdictai/ingestion/parser/` — DoclingParser from `checklist.md` P1, and mark it only if verified working.

Before implementing, read:

1. `AGENTS.md`
2. `checklist.md` P1 section
3. `docs/status.md`
4. `verdictai/ingestion/store/qdrant_store.py`
5. `verdictai/ingestion/store/config.py`
6. Existing test patterns in `tests/test_openai_embedding_provider.py` and `tests/test_qdrant_store_hybrid.py`

## Current Facts

`docs/` did not exist before this planning pass.

Tests currently include `tests/test_openai_embedding_provider.py` and shared fixtures in `tests/conftest.py`. P0 test subdirectories now exist under `tests/retrieval`, `tests/generation`, `tests/serving`, and `tests/eval`, with `.gitkeep` files so the empty structure is preserved.

CUDA is not available in the current environment, so P0 selected the CPU runtime dependency `llama-cpp-python`. `pyproject.toml` now includes all P0 dependencies: `fastembed`, `qdrant-client[fastembed]`, `sentence-transformers`, `tiktoken`, FastAPI, Uvicorn, Redis, Langfuse, Prometheus client, `llama-cpp-python`, `outlines`, `evaluate`, and `bert-score`.

`QdrantStore` now creates dense plus sparse named vectors, computes explicit BM25 sparse vectors with `fastembed.SparseTextEmbedding`, upserts both dense and sparse vectors in each `PointStruct`, and exposes `hybrid_search(dense_vec, sparse_indices, sparse_values, top_k, filters)` using `query_points`, `Prefetch`, `FusionQuery`, and `Fusion.RRF`.

Retrieval, generation, serving, observability, and top-level `eval/` packages are not present yet.

`Makefile` no longer references stale LexRAG or top-level ingestion paths. The `ingest` and `serve` targets now fail explicitly until the corresponding VerdictAI CLI/API surfaces are implemented.

`prompts.md` contains stale guidance for LexRAG, Elasticsearch, BM25-store, and top-level modules. Do not use it as implementation authority.

`docs/status.md` is the current-state handoff document.

## Phase Status

| Phase | Status | Notes |
|---|---|---|
| P0 Foundation | Complete | Core config/logger/pyproject/Makefile exist; P0 dependencies, fastembed verification, test folders, and shared fixtures are done. |
| P1 Ingestion | Partial | Parser, chunker, embeddings, and basic Qdrant store exist; P1-EXT implementation, focused tests, and checklist commit are complete. |
| P1 Eval | Not started | Create top-level `eval/` package after store/retrieval contracts are clear. |
| P2 Retrieval | Not started | Build under `verdictai/retrieval/`. |
| P3 Generation | Not started | Build under `verdictai/generation/`. |
| P4 Serving | Not started | Build under `verdictai/serving/` and `verdictai/observability/`. |
| P5 Frontend/Deploy | Not started | Start only after API works locally. |

## Working Rules For Next Session

Work on one checklist task per pass. Do not mark `[x]` until verification passes. Do not create parallel top-level packages. Do not add LangChain, LlamaIndex, Elasticsearch, or OpenAI for core RAG. Use `UV_CACHE_DIR=/private/tmp/uv-cache` with `make format`, `make lint`, and `make test`.

## Latest Completed Task

Task: Repository alignment cleanup pass.

Files changed:
- `docker-compose.local.yaml`
- `Makefile`
- `docs/status.md`

Commands run:
- `rg -n "elasticsearch|kibana|lexrag|ingestion/pipeline" docker-compose.local.yaml Makefile` returned no matches.
- `docker compose -f docker-compose.local.yaml config` passed.
- `make -n ingest` showed the explicit "not implemented yet" ingestion message and `false`.
- `make -n serve` showed the explicit "not implemented yet" serving message and `false`.

Blockers: none found. Risky or speculative cleanup was skipped, including parser/chunker rewrites and notebook cleanup.

Previous completed task: Commit completed P1-EXT Extend QdrantStore for Hybrid Search work.

## Previous Completed Task

Task: Commit completed P1-EXT Extend QdrantStore for Hybrid Search work.

Files changed:
- `checklist.md`
- `docs/status.md`

Commands run:
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra dev pytest tests/test_qdrant_store_hybrid.py -m integration -v` passed: 2 tests.
- `git status --short` showed unrelated pre-existing modified/untracked files, so only the P1-EXT store files and handoff files were staged.
- `git commit -m "feat(store): extend QdrantStore with sparse vectors and hybrid search"` passed.

Blockers: none. The working tree still contains unrelated pre-existing changes that were intentionally not included in the P1-EXT commit.

Previous completed task: P1-EXT Extend QdrantStore for Hybrid Search.

## Previous Completed Task

Task: P1-EXT Extend QdrantStore for Hybrid Search.

Files changed:
- `verdictai/ingestion/store/qdrant_store.py`
- `tests/test_qdrant_store_hybrid.py`
- `checklist.md`
- `docs/status.md`

Commands run:
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra dev pytest tests/test_qdrant_store_hybrid.py -m integration -v` passed: 2 tests.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra dev ruff check verdictai/ingestion/store/qdrant_store.py tests/test_qdrant_store_hybrid.py` passed.
- `UV_CACHE_DIR=/private/tmp/uv-cache make lint` passed.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra dev pytest tests/` passed: 6 tests.

Blockers: no implementation blockers. The requested checklist commit was not created in this pass; the worktree already had unrelated pre-existing changes, so the commit checkbox remains unchecked.

Previous completed task: Complete all remaining P0 infrastructure items.

## Previous Completed Task

Task: Complete all remaining P0 infrastructure items.

Files changed:
- `pyproject.toml`
- `uv.lock`
- `checklist.md`
- `docs/status.md`
- `tests/conftest.py`
- `tests/retrieval/.gitkeep`
- `tests/generation/.gitkeep`
- `tests/serving/.gitkeep`
- `tests/eval/.gitkeep`

Commands run:
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run python -c "import torch; print(torch.cuda.is_available())"` returned `False`, so `llama-cpp-python` was selected instead of `vllm`.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv add llama-cpp-python outlines evaluate bert-score` passed after network approval.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv lock --check` passed.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run python -c "from fastembed import SparseTextEmbedding; print('ok')"` passed.
- `/Users/ayushsolanki/Desktop/Projects/verdictai/.venv/bin/python -c "from fastembed import SparseTextEmbedding; print('ok')"` passed.
- `python -c "from fastembed import SparseTextEmbedding; print('ok')"` failed because plain `python` resolves to `/Users/ayushsolanki/.pyenv/shims/python`, outside the project uv environment.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run python -c "import llama_cpp, outlines, evaluate, bert_score, sentence_transformers; print('ok')"` passed. Matplotlib created a temporary cache because `/Users/ayushsolanki/.matplotlib` was not writable.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra dev ruff check tests/conftest.py` passed.
- `UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra dev pytest tests/` passed: 4 tests.
- `UV_CACHE_DIR=/private/tmp/uv-cache make lint` passed.

Blockers: none for P0. Note that plain `python` is not the project interpreter in this shell; use `uv run python` or `.venv/bin/python` for project dependency checks.

Previous completed task: P0 add `fastapi`, `uvicorn`, `redis`, `langfuse`, and `prometheus-client` to `pyproject.toml` dependencies.

Update this file after each completed task with the new resume point, test commands, and any discovered blockers.
