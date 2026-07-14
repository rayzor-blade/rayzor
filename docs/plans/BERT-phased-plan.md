# BERT in Nue — Phased Plan

Nue's second production architecture. Design rule (2026-07-14 directive): **engines are
selectable tools per model, never gated on one model's quality bar** — Apple's tools where
present, portable paths everywhere else. BERT is the ideal second tenant: encoder-only means
*every* forward is prefill-shaped (batch × seq, compute-bound) — exactly the regime where the
AMX/BNNSGraph engines shine — and there is no KV cache, no decode loop, no causal mask.

## Current inventory (verified 2026-07-14)

| Piece | State |
|---|---|
| `nue/arch/BertArch.hx` | full builder (143 lines): embeddings + N×TransformerBlock + norms; **never validated against a real model** |
| `transformer/{LayerNorm, MultiHeadAttention, GeluFFN, TransformerBlock}.hx` | present (46/72/39/91 lines), untested at model scale |
| `loader/SafetensorsLoader.hx` (295 lines), `ONNXLoader.hx` | present; bert/roberta/deberta-v2 arch mapping exists |
| `ArchRegistry` | `"bert" → BertArch` registered |
| Tokenizers | **BPE only — WordPiece missing** |
| Pooling / embedding head | **missing** |
| GGUF bert-arch metadata keys | **unverified** |
| Padding/attention masks | **missing** (encoder batches need them) |
| Engines | Haxe SDOT (f32-accum, portable), x86 VNNI, Mac AMX f16 (`BNNSMatMul`), BNNSGraph fp16 (probe, 2.1×), ANE/Metal unexplored |

Reference models (small, GGUF + safetensors both available):
`all-MiniLM-L6-v2` (22M — primary correctness target), `bge-small-en-v1.5` (33M),
later `nomic-embed-text` (long-context variant, rotary — stretch).

---

## Phase 0 — Recon & acceptance harness (small)
**Goal:** define "correct" before writing kernels.
- Pull reference models (GGUF F16/Q8_0 + safetensors) and generate **golden vectors**: for a
  fixed 32-sentence set, dump reference embeddings (HF/sentence-transformers or llama.cpp
  `--embedding`) and reference *tokenizations* (WordPiece ids) to `.bin`/`.json` fixtures.
- Read the existing BertArch/TransformerBlock code against the BERT paper + GGUF tensor names;
  list gaps (bias handling, token-type embeddings, pooling).
- Acceptance gates for all later phases: tokenizer ids **exactly** match goldens; embedding
  cosine vs golden ≥ 0.999 (f32) / ≥ 0.99 (f16 engines); throughput reported as sentences/s
  at seq=128 batch=32.

## Phase 1 — Portable correctness (pure Haxe; M1 + NUC identical)
**Goal:** `nue embed "text"` produces golden-matching vectors with zero Apple dependencies.
- **WordPiece tokenizer** (`tokenizer/WordPieceTokenizer.hx`): greedy longest-match `##`
  continuation, `[CLS]/[SEP]/[PAD]/[UNK]`, lowercase/strip-accent options from model config.
  Validate against golden ids (this is where most BERT ports silently break).
- **GGUF bert keys** in the loader (`bert.embedding_length`, `bert.attention.head_count`, …)
  + token-type embeddings; safetensors path exercised as the alternate loader.
- **Encoder forward**: bidirectional MHA (plain `softmax(QKᵀ/√d)V` — no flash needed at
  seq ≤ 512), LayerNorm **with bias**, GELU, learned position embeddings, padding mask
  (additive −inf on padded keys), CLS + mean pooling, optional L2-normalize.
- Wire through `ArchRegistry`; fix whatever the untested code-paths shake out (expect
  resolution-disease-class compiler findings — the matrix/N2 suites are the regression nets).
- **Exit:** goldens pass on M1 *and* NUC with the pure-Haxe engine; `nue/examples/bert-embed`
  smoke committed.

## Phase 2 — Serving + demo (the product surface)
**Goal:** make it usable and measurable.
- `nue/examples/embed-server`: TCP batch endpoint (sibling of llama-chat-server), batched
  encode (pad-to-bucket), `bench_embed.sh` (sentences/s, p95 latency; cooled runs).
- **Vector-search demo**: embed a document corpus, cosine top-k query — the vector-graph-DB
  story on our own stack (llama-chat generates, embed-server retrieves).
- **Exit:** end-to-end demo + bench baselines recorded (pure-Haxe numbers on both boxes).

## Phase 3 — Apple fast path, near-free (existing AMX f16 engine)
**Goal:** reuse what's shipped. Encoder GEMMs are all batch≥16, so the existing
`RZT_AMX_PREFILL` route (BNNSMatMul f16, f32 accumulate, workspace-tuned) applies as-is.
- Ensure BERT's Linear path routes through the same gate (F16/F32 weights: skip Q4 dequant,
  cache f16 weights directly — simpler than the llama case).
- A/B on the Phase-2 bench: expect ~2× over pure-Haxe f32 on M1.
- **Exit:** engine flag selects {haxe, amx-f16} per run; both pass goldens (f16 gate 0.99).

## Phase 4 — BNNSGraph whole-encoder engine (+ first ANE numbers)
**Goal:** the 2.1×-class engine, applied where it's strongest — a whole fused encoder.
- Author the **entire BERT encoder as one mlprogram** (per seq-bucket: 128/256/512): small
  models make baked weights cheap (~45 MB for MiniLM f16), fixed shapes suit CoreML exactly.
  Authoring script per model (coremltools on python≤3.13; landmines already documented in
  `bnns_graph_bench.rs`: `_v2` link names, alphabetized args, FLOAT32 io default, message-log
  callback for errors).
- Execute via BNNSGraph CPU (`RZT_EMBED_ENGINE=graph`); quality gate = cosine ≥ 0.99 vs golden
  (fp16 accumulation is fine for embeddings — similarity is the metric that matters).
- **Flip `ComputeUnit` to ALL** on the same artifact → first ANE measurements in the project
  (CoreML runtime instead of BNNSGraph for that variant if needed).
- **Exit:** engine matrix {haxe, amx-f16, graph-cpu, coreml-ane} benchmarked on one chart;
  memory updated with the rates.

## Phase 5 — NUC/x86 acceleration (portable parity)
**Goal:** the non-Apple half of the directive.
- Q8_0 GGUF path: int8 encoder GEMM through the existing VNNI `dotI8I7` kernels (weights
  Q8, activations Q8-per-block — same machinery as llama's band, minus super-block scales).
- Threaded batch encode across the SpinPool (encoder parallelism is embarrassing:
  per-sentence and per-band).
- **Exit:** NUC sentences/s recorded; M1-vs-NUC parity table; goldens pass (int8 gate 0.985).

## Phase 6 — Productize the engine selector + second encoder
**Goal:** cement the flexibility directive in config, not env vars.
- Per-model engine selection in the model/run config (`engine = auto|haxe|amx|graph|ane`),
  `auto` = best-available-for-platform with portable fallback; run_bundle/bench echo it.
- Second encoder variant to prove generality: `bge-m3`-class or a cross-encoder reranker
  (same blocks, pair-input) — whichever the vector-search demo needs first.
- Docs: `docs/nue-engines.md` (capability matrix, accumulation properties, platform notes).

---

## Risks / landmines (named up front)
- **WordPiece correctness** — the classic silent-wrongness source; goldens in Phase 0 are the
  defense, exact-id match required.
- **GGUF bert metadata variants** (bge vs nomic vs MiniLM conversions differ in key spelling
  and pooling metadata) — validate per reference model.
- **Untested existing code** — BertArch/MHA/GeluFFN have never run at model scale; expect
  cross-module resolution-class compiler findings; keep repro files, fix compiler-first per
  house rule.
- **Padding masks × engines** — the graph engine wants fixed shapes; bucket + mask rather
  than dynamic shapes in v1.
- **coremltools** pinned to python ≤ 3.13 (BlobWriter breaks on 3.14).

Sequencing: 0 → 1 → 2 ship the portable product; 3 is nearly free after 2; 4 and 5 are
independent of each other (Apple vs NUC tracks) and can interleave; 6 closes the loop.
