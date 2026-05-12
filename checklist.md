# CHECKLIST.md — VerdictAI Build Tracker

> Your Jira. One task = one ticket.
> ✅ = already done in existing code. Skip these.
> [ ] = needs to be built.
> Mark [x] when done. Commit after each section.
>
> Tell Codex: "Implement CHECKLIST task R1: Extend QdrantStore for hybrid search"

---

## PHASE 0 — Foundation ✅ Mostly Done

### P0: Infrastructure
- [x] Docker compose with Qdrant, Redis
- [x] `verdictai/config/settings.py` — AppSettings with Pydantic
- [x] `verdictai/utils/logger.py` — project logger
- [x] `pyproject.toml` with uv
- [x] Makefile basics
- [x] Add `fastembed` to pyproject.toml deps: `uv add fastembed`
- [x] Add `fastapi`, `uvicorn`, `redis`, `langfuse`, `prometheus-client` to pyproject.toml
- [x] Add `vllm` (if GPU) or `llama-cpp-python` (if CPU) to pyproject.toml
- [x] Add `sentence-transformers`, `outlines`, `evaluate`, `bert-score` to pyproject.toml
- [x] Verify: `python -c "from fastembed import SparseTextEmbedding; print('ok')"`
- [x] Create `tests/` folder structure: `mkdir -p tests/retrieval tests/generation tests/serving tests/eval`
- [x] Create `tests/conftest.py` with shared fixtures (sample_chunk, sample_metadata, sample_pages)

---

## PHASE 1 — Ingestion ✅ Mostly Done

### P1: Existing (Do Not Rewrite but do check for quality and add missing stuff only if needed else just check mark these)
- [ ] `verdictai/ingestion/parser/` — DoclingParser
- [ ] `verdictai/ingestion/chunker/` — hierarchical + hybrid chunkers
- [ ] `verdictai/ingestion/embeddings/` — Embedder + providers (OpenAI, SentenceTransformers)
- [ ] `verdictai/ingestion/store/generic_store.py` — abstract base
- [ ] `verdictai/ingestion/store/qdrant_store.py` — basic Qdrant store

### P1-EXT: Extend QdrantStore for Hybrid Search
> **Codex task:** "Read verdictai/ingestion/store/qdrant_store.py.
> Extend it — do not rewrite — to support sparse vectors alongside dense.
> Follow CODEX.md section 4 for the exact Qdrant API calls."

- [x] Add `fastembed.SparseTextEmbedding` import and model init (`Qdrant/bm25`) to `qdrant_store.py`
- [x] Update `ensure_collection()` (or equivalent) to create collection with both `vectors_config` (dense, size=1024) and `sparse_vectors_config` (sparse, BM25)
- [x] Add `_compute_sparse(text: str) -> tuple[list[int], list[float]]` private method — returns (indices, values) for one text string
- [x] Update `upsert()` or equivalent to include both `dense` and `sparse` vectors in each PointStruct
- [x] Add `hybrid_search(dense_vec, sparse_indices, sparse_values, top_k, filters) -> list[ScoredPoint]` method using Qdrant's `query_points` with `FusionQuery(fusion=Fusion.RRF)`
- [x] Write `tests/test_qdrant_store_hybrid.py` — mark `@pytest.mark.integration`
- [x] Test: upsert 3 chunks with both vectors → hybrid_search returns them
- [x] Test: upsert idempotent — same chunk_id upserted twice → collection count unchanged
- [x] Run: `pytest tests/test_qdrant_store_hybrid.py -m integration -v`
- [x] Commit: `feat(store): extend QdrantStore with sparse vectors and hybrid search`

### P1-EVAL: Eval Dataset Setup
> **Codex task:** "Create eval/metrics.py and eval/dataset/qa_pairs.json.
> Follow CODEX.md docstring rules."

- [ ] Create `eval/__init__.py`
- [ ] Create `eval/metrics.py` with `mrr_at_k`, `recall_at_k`, `ndcg_at_k` — Google docstrings, pure functions
- [ ] Create `eval/dataset/qa_pairs.json` — 10 seed QA pairs (pull from CUAD, see PROMPTS.md for format)
- [ ] Create `eval/run_eval.py` — CLI with `--split` (ci=first 50, full=all) and `--retriever` flag
- [ ] Create `tests/eval/test_metrics.py` — unit tests for all 3 metric functions with known inputs
- [ ] Run: `pytest tests/eval/test_metrics.py -v` → all pass
- [ ] Commit: `feat(eval): add retrieval metrics and seed QA pairs`

---

## PHASE 2 — Retrieval `Week 1`

### R1: Hybrid Retriever
> **Codex task:** "Read CODEX.md section 4 (Qdrant hybrid search pattern).
> Create verdictai/retrieval/hybrid_retriever.py"

- [ ] Create `verdictai/retrieval/__init__.py`
- [ ] Create `verdictai/retrieval/hybrid_retriever.py`
- [ ] `HybridRetriever` class — Google class docstring with Attributes section
- [ ] `__init__(self, qdrant_store: QdrantStore, sparse_model_name: str = "Qdrant/bm25")` — init sparse model, log model load time
- [ ] `retrieve(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Chunk]`:
  - Embed query dense: use existing embedder from `verdictai/ingestion/embeddings/`
  - Embed query sparse: `sparse_model.embed([query])` → get indices + values
  - Call `qdrant_store.hybrid_search(dense_vec, sparse_indices, sparse_values, top_k*4, filters)`
  - Convert ScoredPoints back to Chunk objects
  - Return top `top_k`
  - Log: query[:50], num_results, elapsed_ms
- [ ] Create `tests/retrieval/test_hybrid_retriever.py`
- [ ] Test (mock): `retrieve()` calls `qdrant_store.hybrid_search` once
- [ ] Test (mock): `retrieve()` returns list of Chunk objects
- [ ] Test (mock): `top_k` is respected in output length
- [ ] Run: `pytest tests/retrieval/test_hybrid_retriever.py -v` → all pass
- [ ] Commit: `feat(retrieval): add HybridRetriever with Qdrant RRF fusion`

### R2: Cross-Encoder Reranker
> **Codex task:** "Create verdictai/retrieval/reranker.py.
> Follow CODEX.md docstring and logging rules."

- [ ] Create `verdictai/retrieval/reranker.py`
- [ ] Constant: `RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"`
- [ ] `CrossEncoderReranker` class — Google class docstring
- [ ] `__init__(self, model_name: str = RERANK_MODEL)` — load CrossEncoder, log load time
- [ ] `rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]`:
  - Create pairs: `[(query, c.text) for c in chunks]`
  - `scores = self.model.predict(pairs)`
  - Sort chunks by score descending
  - Log: score min/max/mean (debug level)
  - Return `chunks[:top_k]`
- [ ] Create `tests/retrieval/test_reranker.py` — mark `@pytest.mark.slow`
- [ ] Test: reranker changes order of 5 chunks vs original order
- [ ] Test: output len == top_k
- [ ] Test: output is subset of input (no new chunks invented)
- [ ] Run: `pytest tests/retrieval/test_reranker.py -m slow -v`
- [ ] Commit: `feat(retrieval): add CrossEncoderReranker`

### R3: Query Decomposer
> **Codex task:** "Create verdictai/retrieval/query_decomposer.py"

- [ ] Create `verdictai/retrieval/query_decomposer.py`
- [ ] `is_multihop(question: str) -> bool` — heuristic: >15 words AND contains "if"/"when"/"after"/"provided that"/"and also"
- [ ] `decompose(question: str, llm_fn: Callable[[str], str]) -> list[str]`:
  - Prompt: `"Break this legal question into at most 3 simpler sub-questions. Return JSON list of strings only. Question: {question}"`
  - Parse JSON response, cap at 3 sub-questions, return list
- [ ] Create `tests/retrieval/test_query_decomposer.py`
- [ ] Test: `is_multihop("What is the liability cap if breach occurs after warranty and governing law is Delaware?")` → True
- [ ] Test: `is_multihop("What is the liability cap?")` → False
- [ ] Test: `decompose()` returns list of max 3 strings (mock the llm_fn)
- [ ] Run: `pytest tests/retrieval/test_query_decomposer.py -v`
- [ ] Commit: `feat(retrieval): add query decomposer for multi-hop questions`

### R4: Phase 2 Eval
- [ ] Expand `eval/dataset/qa_pairs.json` to 50 questions (20 factoid, 15 multi_hop, 10 unanswerable, 5 temporal) — pull from CUAD dataset
- [ ] Run: `python eval/run_eval.py --split full --retriever hybrid`
- [ ] Save to `eval/results/phase2_baseline.json`
- [ ] Record MRR@5 number — write it down. This is your baseline.
- [ ] Create `docs/ablation.md` with Phase 2 baseline row
- [ ] Commit: `experiment(eval): phase 2 baseline — MRR@5: X.XX`

---

## PHASE 3 — Generation `Week 2`

### G1: LLM Generator
> **Codex task:** "Create verdictai/generation/generator.py.
> Auto-detect GPU vs CPU. Follow CODEX.md rules."

- [ ] Create `verdictai/generation/__init__.py`
- [ ] Create `verdictai/generation/generator.py`
- [ ] Constant: `LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"`
- [ ] `LLMGenerator` class — Google class docstring
- [ ] `__init__(self)`:
  - Check `torch.cuda.is_available()`
  - GPU: init vLLM `AsyncLLMEngine`
  - CPU: init `llama_cpp.Llama` with GGUF path from settings
  - Log which path was chosen + load time
- [ ] `generate(self, prompt: str, max_tokens: int = 800) -> tuple[str, list[float]]`:
  - Returns `(answer_text, logprobs)` — logprobs needed for uncertainty scoring
  - Log: prompt token count, completion token count, time_to_first_token_ms
- [ ] Create `tests/generation/test_generator.py`
- [ ] Test (mock model): `generate()` returns tuple of (str, list)
- [ ] Test: context budget respected — mock with 5000-token prompt, assert it gets trimmed
- [ ] Run: `pytest tests/generation/test_generator.py -v`
- [ ] Commit: `feat(generation): add LLMGenerator with GPU/CPU auto-detection`

### G2: Prompt Builder
> **Codex task:** "Create verdictai/generation/prompt_builder.py"

- [ ] Create `verdictai/generation/prompt_builder.py`
- [ ] `SYSTEM_PROMPT` constant — legal assistant, cite every claim as `[SOURCE: chunk_id, p.N]`, abstain if unsure
- [ ] `MAX_CONTEXT_TOKENS = 3800` constant
- [ ] `build_prompt(question: str, chunks: list[Chunk]) -> str`:
  - Format each chunk: `[{chunk_id} | {doc_type} | p.{page_num}]\n{text}`
  - Count tokens (use tiktoken cl100k_base)
  - If over budget: drop last chunk, repeat until under budget
  - Return full prompt string
- [ ] `count_tokens(text: str) -> int` helper
- [ ] Create `tests/generation/test_prompt_builder.py`
- [ ] Test: all chunk_ids appear in output
- [ ] Test: token budget respected with 10 large chunks
- [ ] Run: `pytest tests/generation/test_prompt_builder.py -v`
- [ ] Commit: `feat(generation): add prompt builder with token budget management`

### G3: Citation Formatter
> **Codex task:** "Create verdictai/generation/citation_formatter.py"

- [ ] Create `verdictai/generation/citation_formatter.py`
- [ ] `CitationFormattedAnswer` dataclass: `answer_text`, `claims` (list[str]), `citations` (list[dict]), `has_hallucinated_citations` (bool)
- [ ] `CitationFormatter` class
- [ ] `parse_citations(answer_text: str, available_chunk_ids: list[str]) -> dict`:
  - Regex: find all `[SOURCE: {chunk_id}, p.{N}]` patterns
  - Flag any chunk_id not in `available_chunk_ids` as hallucinated
  - Return `{claims, citations, hallucinated}`
- [ ] `format_answer(raw_answer: str, chunks: list[Chunk]) -> CitationFormattedAnswer`
- [ ] Create `tests/generation/test_citation_formatter.py`
- [ ] Test: answer with fake chunk_id → `has_hallucinated_citations=True`
- [ ] Test: answer with real chunk_id → in citations list
- [ ] Test: no citations in answer → empty list, not error
- [ ] Run: `pytest tests/generation/test_citation_formatter.py -v`
- [ ] Commit: `feat(generation): add citation formatter and hallucination detection`

### G4: Faithfulness Gate
> **Codex task:** "Create verdictai/generation/faithfulness_gate.py.
> Use cross-encoder/nli-deberta-v3-base. Follow CODEX.md section 5 docstring format."

- [ ] Create `verdictai/generation/faithfulness_gate.py`
- [ ] Constants: `NLI_MODEL = "cross-encoder/nli-deberta-v3-base"`, `NEUTRAL_THRESHOLD = 0.3`
- [ ] `GateResult` dataclass: `passed`, `abstained`, `low_confidence`, `per_claim_scores`, `reason`
- [ ] `FaithfulnessGate` class — Google docstring explaining NLI label order `[contradiction, neutral, entailment]`
- [ ] `check(self, claims: list[str], chunk_map: dict[str, Chunk]) -> GateResult`:
  - For each claim: run NLI model against cited chunk text
  - Contradiction → immediately return `abstained=True`
  - `neutral_count / total > NEUTRAL_THRESHOLD` → `low_confidence=True`
  - All good → `passed=True`
- [ ] Create `tests/generation/test_faithfulness_gate.py` — mark `@pytest.mark.slow`
- [ ] Test: claim contradicts chunk → `abstained=True`
- [ ] Test: claim matches chunk → `passed=True`
- [ ] Test: empty claims → `passed=True` (nothing to gate)
- [ ] Run: `pytest tests/generation/test_faithfulness_gate.py -m slow -v`
- [ ] Commit: `feat(generation): add DeBERTa-v3 NLI faithfulness gate`

### G5: Uncertainty Scorer
> **Codex task:** "Create verdictai/generation/uncertainty.py — small file, pure logic."

- [ ] Create `verdictai/generation/uncertainty.py`
- [ ] `UncertaintyResult` dataclass: `mean_logprob`, `confidence_bucket` (Literal['high','medium','low'])
- [ ] `UncertaintyScorer` class
- [ ] `score(self, logprobs: list[float]) -> UncertaintyResult`:
  - `mean = sum(logprobs) / len(logprobs)` (handle empty: return medium)
  - `> -0.3` → high, `-0.3 to -0.8` → medium, `< -0.8` → low
- [ ] Create `tests/generation/test_uncertainty.py`
- [ ] Test: `[-0.1, -0.2, -0.1]` → `high`
- [ ] Test: `[-1.0, -0.9, -1.2]` → `low`
- [ ] Test: `[]` → `medium` (safe default)
- [ ] Run: `pytest tests/generation/test_uncertainty.py -v`
- [ ] Commit: `feat(generation): add token entropy uncertainty scorer`

### G6: RAG Chain
> **Codex task:** "Create verdictai/generation/rag_chain.py — orchestrates all generation components."

- [ ] Create `verdictai/generation/rag_chain.py`
- [ ] `RAGResponse` dataclass: `answer`, `abstained`, `confidence`, `citations`, `latency_ms`, `reason`
- [ ] `RAGChain` class — Google docstring listing all injected components
- [ ] `answer(self, question: str, filters: Optional[dict] = None) -> RAGResponse`:
  - Step 1: `chunks = retriever.retrieve(question, filters=filters)` — time it
  - Step 2: if no chunks → return abstained `RAGResponse`
  - Step 3: `prompt = prompt_builder.build_prompt(question, chunks)`
  - Step 4: `raw_answer, logprobs = generator.generate(prompt)` — time it
  - Step 5: `formatted = citation_formatter.format_answer(raw_answer, chunks)`
  - Step 6: `gate_result = faithfulness_gate.check(formatted.claims, chunk_map)` — time it
  - Step 7: `uncertainty = uncertainty_scorer.score(logprobs)`
  - Step 8: if abstained → return abstained `RAGResponse`
  - Step 9: return full `RAGResponse` with all fields
  - Log: question[:60], num_chunks, confidence, total_ms, abstained
- [ ] Create `tests/generation/test_rag_chain.py` — mark `@pytest.mark.integration @pytest.mark.slow`
- [ ] Test (all mocked): answer returned has citations field populated
- [ ] Test (all mocked): unanswerable → abstained=True
- [ ] Test: latency_ms dict has all expected keys
- [ ] Run: `pytest tests/generation/test_rag_chain.py -v` (mocked version, no real models)
- [ ] Commit: `feat(generation): add RAGChain orchestrator`

### G7: Phase 3 Eval
- [ ] Add `faithfulness_score`, `bertscore_f1`, `citation_accuracy` to `eval/metrics.py`
- [ ] Update `eval/run_eval.py` to run full pipeline (retrieval + generation)
- [ ] Expand `eval/dataset/qa_pairs.json` to 100 questions
- [ ] Run: `python eval/run_eval.py --split full`
- [ ] Save to `eval/results/phase3_full.json`
- [ ] Verify faithfulness > 0.80 (if not, check NLI model loaded correctly)
- [ ] Run gate ablation: with/without faithfulness gate — record delta
- [ ] Save to `eval/results/phase3_gate_ablation.json`
- [ ] Update `docs/ablation.md`: Phase 3 rows
- [ ] Create `docs/failures.md` — write 3 real failure cases you observed
- [ ] Create `docs/decisions.md` — ADR-001 (Qdrant hybrid, no ES), ADR-002 (faithfulness gate)
- [ ] Commit: `experiment(eval): phase 3 full eval — faithfulness: X.XX`

---

## PHASE 4 — Serving `Week 3`

### S1: FastAPI App + Models
> **Codex task:** "Create verdictai/serving/app.py and verdictai/serving/models.py.
> Models must load in lifespan, never in handlers. Follow CODEX.md section 6."

- [ ] Create `verdictai/serving/__init__.py`
- [ ] Create `verdictai/serving/models.py` — all Pydantic models with docstrings:
  - `QueryRequest`: question, filters, stream (default True)
  - `CitationResponse`: chunk_id, doc_id, page_num, section_title, excerpt (200 chars)
  - `LatencyBreakdown`: retrieval_ms, rerank_ms, generation_ms, total_ms
  - `QueryResponse`: answer, abstained, confidence, citations, latency, trace_id
  - `IngestResponse`: doc_id, chunks_created, time_ms
- [ ] Create `verdictai/serving/routes/__init__.py`
- [ ] Create `verdictai/serving/app.py`:
  - `lifespan()` async context manager — loads all models into `app.state` on startup
  - `app = FastAPI(title="VerdictAI", version="1.0.0", lifespan=lifespan)`
  - `GET /` → `{"status": "ok"}`
- [ ] Test: `uvicorn verdictai.serving.app:app --reload` starts without error
- [ ] Commit: `feat(serving): add FastAPI app with lifespan model loading`

### S2: /query Endpoint
> **Codex task:** "Create verdictai/serving/routes/query.py"

- [ ] Create `verdictai/serving/routes/query.py`
- [ ] `POST /query` endpoint:
  - Check Redis cache first → return cached if hit
  - Call `app.state.rag_chain.answer(request.question, filters=request.filters)`
  - Build `QueryResponse` from `RAGResponse`
  - Add `trace_id = str(uuid4())`
  - Cache result if not abstained
  - Streaming: if `request.stream=True` → `StreamingResponse` yielding tokens as SSE
- [ ] Include router in `app.py`
- [ ] Test: `curl -X POST http://localhost:8000/query -d '{"question":"test","stream":false}'` returns JSON
- [ ] Commit: `feat(serving): add /query endpoint with streaming and caching`

### S3: /ingest + /health Endpoints
> **Codex task:** "Create verdictai/serving/routes/ingest.py and health.py"

- [ ] Create `verdictai/serving/routes/ingest.py` — `POST /ingest` with file upload → IngestPipeline
- [ ] Create `verdictai/serving/routes/health.py` — `GET /health` pings Qdrant + Redis, returns status
- [ ] Include both in `app.py`
- [ ] Test: `GET /health` → `{"qdrant": true, "redis": true}`
- [ ] Commit: `feat(serving): add /ingest and /health endpoints`

### S4: Redis Semantic Cache
> **Codex task:** "Create verdictai/serving/cache.py"

- [ ] Create `verdictai/serving/cache.py`
- [ ] `SemanticCache` class — Google docstring
- [ ] `get(question: str) -> Optional[QueryResponse]`:
  - Embed question
  - Fetch cached embeddings from Redis
  - Cosine sim > 0.95 → return cached response
  - Else → None
- [ ] `set(question: str, response: QueryResponse, ttl: int = 86400)`:
  - Store embedding + response JSON
  - Never cache abstained responses
- [ ] Create `tests/serving/test_cache.py`
- [ ] Test (mock Redis): same question → cache hit on second call
- [ ] Test: abstained response not cached
- [ ] Run: `pytest tests/serving/test_cache.py -v`
- [ ] Commit: `feat(serving): add Redis semantic cache`

### S5: Auth + Rate Limiting
> **Codex task:** "Create verdictai/serving/middleware.py"

- [ ] Create `verdictai/serving/middleware.py`
- [ ] API key auth: read `X-API-Key` header, check against `settings.api_secret_key`, return 401
- [ ] Rate limiter: Redis key `rate:{api_key}`, max 30/min, return 429 with `Retry-After` header
- [ ] `/health` exempt from auth
- [ ] Add to `app.py`
- [ ] Commit: `feat(serving): add API key auth and rate limiting`

### S6: Observability
> **Codex task:** "Create verdictai/observability/langfuse_client.py and prometheus_metrics.py"

- [ ] Create `verdictai/observability/__init__.py`
- [ ] Create `verdictai/observability/langfuse_client.py`:
  - `LangfuseTracer` class — `start_trace`, `log_retrieval`, `log_generation`, `finish_trace`
  - Calls added to `RAGChain.answer()` at each step
- [ ] Create `verdictai/observability/prometheus_metrics.py`:
  - `verdictai_query_total` Counter (labels: status, confidence)
  - `verdictai_query_latency_ms` Histogram (labels: stage)
  - `verdictai_cache_hits_total` Counter (labels: result)
  - `verdictai_faithfulness_score` Gauge
- [ ] Mount `/metrics` in `app.py`
- [ ] Verify: `curl http://localhost:8000/metrics` → Prometheus text format
- [ ] Commit: `feat(observability): add LangFuse tracing and Prometheus metrics`

### S7: Load Test
- [ ] Install k6: `brew install k6`
- [ ] Create `scripts/load_test.js` — 50 VUs, 60s, 10 random questions
- [ ] Run: `k6 run scripts/load_test.js`
- [ ] Record P50, P95, P99 latency + error rate
- [ ] Add to `docs/ablation.md`
- [ ] If P95 > 2.5s → profile bottleneck, increase cache or reduce max_tokens
- [ ] Commit: `test(serving): load test results — P95: Xms`

---

## PHASE 5 — Frontend + Deploy `Week 4`

### F1: Next.js Setup
- [ ] `cd frontend && npx create-next-app@latest . --typescript --tailwind --app`
- [ ] `npm install @tanstack/react-query lucide-react recharts`
- [ ] Create `.env.local`: `NEXT_PUBLIC_API_URL=http://localhost:8000`
- [ ] Create `lib/api.ts` — streaming fetch wrapper for `/query`
- [ ] Create `types/api.ts` — TypeScript interfaces matching serving/models.py
- [ ] Commit: `feat(frontend): scaffold Next.js app`

### F2: Core UI
- [ ] `components/QueryBox.tsx` — textarea, submit, doc_type + jurisdiction filters
- [ ] `components/AnswerStream.tsx` — streaming token display
- [ ] `components/CitationBadge.tsx` — clickable `[p.N]` badge
- [ ] `components/CitationPanel.tsx` — right drawer with chunk text
- [ ] `components/ConfidenceBadge.tsx` — green/yellow/red dot
- [ ] Wire `app/page.tsx` — QueryBox → stream → Answer + Citations
- [ ] Commit: `feat(frontend): add query interface with streaming and citations`

### F3: Example Questions + Upload
- [ ] `constants/examples.ts` — 6 example questions (factoid, multi-hop, regulatory, EDGAR, abstention, temporal)
- [ ] Show as clickable chips, auto-submit on click
- [ ] `components/Uploader.tsx` — drag-and-drop PDF → POST /ingest → show progress stages
- [ ] Commit: `feat(frontend): add example questions and document upload`

### F4: Ablation Dashboard
- [ ] `app/ablation/page.tsx` — hard-coded ablation data as TypeScript array
- [ ] Recharts bar charts: MRR@5 + faithfulness progression across phases
- [ ] Link from README
- [ ] Commit: `feat(frontend): add ablation results dashboard`

### F5: Deploy
- [ ] Sign up Qdrant Cloud → free cluster → get URL + API key
- [ ] Sign up Upstash → free Redis → get REDIS_URL
- [ ] Sign up Modal Labs → `modal setup` → create `infra/modal_deploy.py` (vLLM T4 GPU)
- [ ] `modal deploy infra/modal_deploy.py` → get LLM endpoint URL
- [ ] Sign up Fly.io → `fly launch` → `fly secrets set ...` → `fly deploy`
- [ ] Verify: `curl https://your-app.fly.dev/health` → all green
- [ ] Deploy frontend: `cd frontend && npx vercel --prod`
- [ ] Test live URL end-to-end
- [ ] Commit: `feat(infra): production deployment on Fly.io + Modal + Vercel`

### F6: Final Docs + Launch
- [ ] Write README sections: problem, demo link, architecture diagram (Mermaid), ablation table, 3 failure cases, stack, run locally (5 commands), cost breakdown
- [ ] Record 5-min demo video → YouTube unlisted → link in README
- [ ] Post LinkedIn launch post (use Post 5 from content plan)
- [ ] Commit: `docs: final README — live at <url>`

---

## Progress

| Phase | Tasks | Done | Remaining |
|---|---|---|---|
| P0 Foundation | 10 | 7 | 3 |
| P1 Ingestion | 15 | 10 | 5 |
| P2 Retrieval | 18 | 0 | 18 |
| P3 Generation | 28 | 0 | 28 |
| P4 Serving | 30 | 0 | 30 |
| P5 Frontend+Deploy | 22 | 0 | 22 |
| **Total** | **123** | **17** | **106** |

> Update Done column at end of each phase.
