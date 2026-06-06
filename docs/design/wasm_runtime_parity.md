# WASM Runtime Parity — Architecture Plan

Status: draft (2026-06-06)
Owners: TBD
Tracking: project_wasm_phase2.md (current Phase 2 work — `_start` arity, integer SIMD)

## Problem statement

The native runtime (`runtime/`) has the full Rayzor + nue ML inference
stack — tensor allocation, Q4_K_M / Q6_K / Q8_0 SDOT matmul kernels,
flash attention, KV cache, RoPE, RMSNorm, softmax, GGUF loading, plus
a worker pool threading model. Llama-3.2-1B Q4_K_M decode runs at
~39 tok/s cool-state on M1 Pro.

The WASM runtime (`runtime-wasm/`) has ~2.6k lines covering Haxe FFI
glue (strings, math, anon objects, Bytes), SIMD4f for vector graphics,
and **four** tensor primitives (`add_f32`, `mul_f32`, `dot_f32`,
`sum_f32`). It does NOT have:

- Tensor allocation / lifetime model
- Quantized matmul (any scheme)
- Flash attention
- KV cache
- RoPE, RMSNorm, softmax
- Sampling, top-k, top-p
- GGUF parsing
- Threading abstraction (the native `worker_pool` uses `std::thread`,
  which doesn't exist on `wasm32-wasip1-threads`)
- Any of the perf optimisations landed this session (parallel flash
  attention over q_heads, thread_local Q8K scratch, USER_INTERACTIVE
  QoS, NEON-specific SIMD inner ops)

Full parity = "Llama-3.2-1B Q4_K_M decode runs in a Chromium tab
within 2-3× of native" + "the API surface that nue/ writes to has
identical behaviour on native and WASM, modulo measurable perf."

This document plans how to get there.

## Goals

- **G1.** WASM target supports the full nue/ ML inference pipeline
  for at least Llama-3.2-1B Q4_K_M. Same Haxe source compiles for
  both targets; same Paris MATCH on canonical prompt.
- **G2.** Threading model works in three deployment contexts:
  (a) `wasm32-wasip1-threads` via `wasmtime` (server-side / CLI),
  (b) browser with COOP/COEP headers + Web Workers + SharedArrayBuffer,
  (c) single-threaded fallback when neither is available (lowest
  common denominator — degrades to single core, doesn't crash).
- **G3.** Code sharing between native and WASM runtimes is structural,
  not copy-paste. A kernel update lands once and propagates.
- **G4.** No regression in native perf or correctness from the
  refactor work.

## Non-goals

- **NG1.** GPU acceleration in browser via WebGPU. That's a separate
  track (`gpu/src/codegen/wgsl.rs` exists; `examples/wasm-features/`
  has demos). This plan covers the CPU runtime.
- **NG2.** Matching llama.cpp browser-side performance. llama.cpp +
  Emscripten + threads currently hits ~10-15 tok/s on M1 Pro in
  Chrome. We aim for "usable" (~5 tok/s) first; perf optimisation
  is a later phase.
- **NG3.** Eliminating the `runtime-wasm/` crate boundary. Pure
  Rust kernels can be shared; the FFI surface (`extern "C"` exports
  the WASM bytecode resolves against) stays per-target because the
  ABI conventions differ.
- **NG4.** Other quantisation schemes beyond what native supports
  today (Q4_K_M, Q6_K, Q8_0). FP8 / new quant formats are scoped
  separately.

## Current architecture

```
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ runtime/                     │    │ runtime-wasm/                │
│ ~30k LOC                     │    │ ~2.6k LOC                    │
│                              │    │                              │
│ tensor.rs   ← flash attn     │    │ lib.rs ── all-in-one         │
│ quant.rs    ← Q4/Q6/Q8 SDOT  │    │                              │
│ worker_pool ← std::thread    │    │ (no threads, no quant,       │
│ tensor_simd ← NEON / SSE2    │    │  no matmul, no flash attn)   │
│ profile.rs  ← SIGPROF        │    │                              │
│ rng / sampling               │    │ Tensor SIMD: only +/×/dot/Σ  │
│ GGUF loader (via plugin)     │    │ on contiguous f32 slices     │
└──────────────────────────────┘    └──────────────────────────────┘
       ↓ link with                          ↓ link as wasm32
   rayzor (host binary)                 .wasm runtime image
   nue-plugins (.dylib)                 (Web/CLI/wasi)
```

The two crates are **independent code paths** with no shared algorithm
implementation. Drift between them is currently policed by tests in
`compiler/tests/haxe/` that pass on both targets — but only for the
APIs that runtime-wasm actually implements.

## Code-sharing options

Three approaches considered. Recommendation: **Option C, phased**.

### Option A — Per-crate ports

Native engineer writes a kernel in `runtime/`; WASM port copy-pastes
into `runtime-wasm/` and adapts for `wasm32` intrinsics. Easiest to
start but creates permanent drift. Already what we have implicitly,
and the gap is the symptom.

### Option B — Cargo cfg gates inside `runtime/`

Make `runtime/` compile for both native AND `wasm32`. Gate
NEON/SSE2 paths behind `#[cfg(target_arch = ...)]`, add `wasm_simd128`
paths.

Pros: one crate, one source of truth.
Cons:
- `runtime/` has many deps that aren't `no_std` / wasm-friendly
  (`libc::malloc` directly, signal handling for SIGPROF profiling,
  thread pools using `std::thread`, libloading for plugins…).
- Conditional compilation across the entire crate quickly becomes
  unreadable.
- The wasm32 build would be all-or-nothing — can't pick "just the
  kernels".

### Option C — Extract `runtime-core/` ← recommended

Pull the *algorithmic kernels* (pure compute, no syscalls, no
threading, no allocator coupling) into a new crate `runtime-core/`
that:

- Is `no_std + alloc`-friendly (works for both targets).
- Has `cfg(target_feature = ...)` SIMD specialisations: aarch64
  NEON, x86_64 SSE2/AVX2, wasm32 simd128 + relaxed-simd, scalar.
- Has NO global state, NO threading. Takes pre-allocated buffers
  and operates on slices.
- Includes the SDOT-style quant kernels written portably: scalar
  reference + per-arch intrinsics behind cfg gates.

Then:

- `runtime/` (native) depends on `runtime-core`. Adds threading via
  `std::thread` worker pool, SIGPROF profiling, libloading, etc.
- `runtime-wasm/` depends on `runtime-core`. Adds wasi-thread bindings
  (server-side) or stays single-threaded (browser fallback). The
  WASM-specific orchestration (sleep, allocator-via-imports, etc.)
  stays here.

The FFI surface (`extern "C"` exports the JIT bytecode resolves
against) is implemented in the per-target crate; the *body* of each
exported function is a thin wrapper around a `runtime-core` call.

Estimated cost:
- Initial extraction: 2-3 days. Touches `quant.rs`, `tensor.rs`,
  `tensor_simd.rs`, `flash_attn` helpers — ~3-5k LOC moved.
- Per-kernel migration: incremental. New kernels go straight into
  `runtime-core`; existing kernels migrate when touched.

This is the only path that gets us G3 (structural sharing).

## Threading

Three target environments, three different threading stories.

### 3a. `wasm32-wasip1-threads` (CLI / `wasmtime`)

- WASI thread API specced by `wasi-threads-spec`. Implemented by
  `wasmtime` 13+ behind `--wasi=preview1,threads`.
- `runtime-wasm/.cargo/config.toml` already enables `+atomics`,
  `--shared-memory`, `--import-memory`. The bytecode declares
  shared linear memory; the embedder hooks up Web Worker or
  pthread-equivalent.
- `std::thread::spawn` on `wasm32-wasip1-threads` lowers to
  `wasi_thread_spawn` automatically when std is built with the
  threads feature.
- Worker pool from `runtime-core` should work without changes —
  the abstraction can use `std::thread::spawn` and the wasm32-wasip1-threads
  build picks up the WASI implementation.

This is the easy path. `wasmtime --wasi=threads-yes server.wasm`
should give native-like threading.

### 3b. Browser — COOP/COEP gated SharedArrayBuffer

- **The hard constraint.** Browsers disable `SharedArrayBuffer`
  without:
  - `Cross-Origin-Opener-Policy: same-origin` on the document
  - `Cross-Origin-Embedder-Policy: require-corp` on the document
  - All cross-origin subresources must also opt in via
    `Cross-Origin-Resource-Policy: cross-origin` or
    `Cross-Origin-Embedder-Policy: require-corp`
- Without SharedArrayBuffer:
  - `wasm32-wasip1-threads` modules instantiate but threads fail
    at `wasi_thread_spawn` time
  - No `Atomics.wait` (its spec requires SAB)
  - No way to share memory between Web Workers
  - → multithreading is impossible. Single-threaded fallback only.

- **With SharedArrayBuffer enabled**, the wiring is:
  - Main thread owns the WASM module + shared memory + worker pool
  - Each worker is a `Web Worker` that imports the same WASM and
    receives the shared memory via `postMessage`
  - The runtime-wasm worker_pool::parallel_rows pumps work into a
    SharedArrayBuffer-backed task queue; workers `Atomics.wait` /
    `Atomics.notify` to coordinate
  - Per-thread state (e.g. our new `thread_local!` Q8K scratch) is
    per-Web-Worker, exactly as on native
  - Memory budget for browser tabs is typically 2-4 GB; matches the
    `--max-memory=1073741824` (1 GiB) we already declare

- **Implementation effort**: a JS-side `rayzor-worker-host.js` glue
  module (~300 LOC) that hosts the Web Worker pool, plus a Rust-side
  `wasm32` cfg path in `runtime-core/worker_pool.rs` that calls into
  JS imports for `spawn_worker(fn_idx)`, `notify(addr)`,
  `wait(addr, expected)`.

### 3c. Single-threaded fallback

When neither (3a) nor (3b) is available:

- The runtime detects this at init (via JS feature probe in browser,
  or a build-time cfg in `wasi`).
- The worker pool turns into a no-op: `parallel_rows(n, t, f)`
  becomes `f(0, n)` on the main thread.
- The parallelisation gate (e.g. flash_attn_decode's `cache_len >=
  256` check) just doesn't fire because `t = 1`.
- All correctness is preserved; only perf suffers.

This is the path that runs anywhere — file://, GitHub Pages with no
custom headers, embedded in third-party sites — and is critical for
the demo experience. The plan is to ship single-threaded WORKING
before multi-threaded FAST.

## COOP/COEP — deployment story

We need both server-side (runtime headers) and client-side (origin
attribution) work. Concrete deployment paths:

### Self-hosted (full control)

Deploy with these headers on the HTML:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Plus the WASM artefacts served with:

```
Cross-Origin-Resource-Policy: cross-origin
```

Use case: rayzor.dev demo, internal company deployments. Maximum perf
available.

### GitHub Pages / Cloudflare Pages / Netlify

GitHub Pages doesn't allow custom response headers. Workarounds:

- **`coi-serviceworker`** trick: register a service worker that
  intercepts top-level navigations and rewrites the response to
  include COOP/COEP. This works because the service worker runs
  in the page's origin and can rewrite its OWN responses. There
  is a maintained NPM package (`coi-serviceworker`) we can vendor
  into the deployable artefact.
- Approach: ship a small JS bundle that registers the SW on
  first visit (reload required) then loads rayzor. Single-threaded
  on first visit, multi-threaded on second visit.

Use case: zero-infra demos. Worth documenting + maintaining; not
worth blocking the project on.

### Embedded in third-party sites

When rayzor is embedded in an `<iframe>` on a host page we don't
control:

- Either the host page sets COOP/COEP (their problem)
- Or rayzor runs single-threaded (our fallback covers this)

We can't unilaterally make this work. Don't promise embedded
multi-threading.

### Local dev (`http://localhost`)

For day-to-day development:

- The dev server (whatever Vite / Webpack / esbuild config the
  examples use) needs to set COOP/COEP. We ship a sample server
  config in `examples/wasm-features/`.
- Browsers like Chrome have `chrome://flags/#enable-features=` toggle
  but we shouldn't rely on it.
- Cleanest: a Rust-based dev server in `tools/wasm-dev-serve/` that
  serves with the right headers. 50 LOC with `axum` or `tide`.

## Phased roadmap

Each phase ends with: working build + Paris MATCH on the relevant test
+ a perf number where applicable.

### Phase 1 — `runtime-core` extraction (foundation)

**Goal:** code-sharing infrastructure ready, no behavioural change.

- Create `runtime-core/` crate. `no_std + alloc`. Move:
  - `tensor_simd.rs` (existing aarch64/x86_64 paths + scalar
    fallback + new `cfg(target_feature = "simd128")` paths
    using `core::arch::wasm32`)
  - The pure-compute helpers from `quant.rs`: `quantize_q8_k_block`,
    `dequant_q4_k_block`, `dequant_q6_k_block`, `vec_dot_q4_k_q8_k`,
    `vec_dot_q6_k_q8_k`, etc.
  - The pure-compute helpers from `tensor.rs`: `softmax_inplace`,
    `rope_apply`, `rmsnorm_apply`, and the inner body of
    `flash_attn_decode_one_qhead`.
- Native `runtime/` re-exports the same symbols via thin wrappers
  that handle the `extern "C"` boundary + global state.
- `runtime-wasm/` deps unchanged until Phase 2.

**Verification:** all native examples + Paris MATCH unchanged. Llama
decode tok/s unchanged within noise. Workspace strict-mode clippy
remains clean.

**Effort:** 2-3 days focused.
**Risk:** Medium — touching `runtime/` is the kind of refactor that
this session showed can introduce subtle bugs. Mitigation: incremental
commits, verify Paris MATCH after each crate-extraction step.

### Phase 2 — WASM tensor allocation + F32 matmul

**Goal:** `runtime-wasm` has a Tensor type and a working F32 matmul.

- Port `RayzorTensor` struct + alloc/free path from `runtime/` to
  `runtime-wasm/`. Memory comes from `dlmalloc` (already used for
  Rust's `wasm32` allocator) so no allocator engineering needed.
- Port `matmul_t_f32` (the F32 unfused path) using `runtime-core`
  inner kernels.
- Wire to Haxe: `Tensor.zeros`, `Tensor.fromFloats`,
  `Tensor.matmulT` as MIR wrappers calling the new wasm-runtime
  symbols.
- Build for `wasm32-wasip1-threads`; run a smoke test via
  `wasmtime` (no nue yet; just F32 matmul correctness).

**Verification:** `tests/wasm/` smoke that builds a 64x64 F32 matmul
and matches the native result within FP32 ULP tolerance.

**Effort:** 3-4 days.

### Phase 3 — Quantised matmul (Q8_0, then Q4_K_M)

**Goal:** decode-grade quantised matmul on WASM.

- Q8_0 first because it's the simplest layout. Pure-Rust port (the
  NEON SDOT path is hardware-specific; on wasm32 we use scalar +
  `f32x4_relaxed_madd` for the f32 accumulator side).
- Q4_K_M next. The K-quant inner loop is structurally identical
  across hardware; only the SDOT instruction changes. On wasm32 we
  use scalar i8-multiply-accumulate (no `i8x16.dot_i8x16_i7x16` yet
  in stable WASM — keep an eye on the proposal).
- Q6_K third.
- Each kernel comes with its `runtime-core` scalar + wasm SIMD
  implementation; the native crate's NEON path stays separate.

**Verification:** Q8_0 / Q4_K_M / Q6_K dequant-then-matmul vs the
fused matmul match within FP tolerance.

**Effort:** 4-7 days per scheme. Q4_K_M is the work-horse — invest in
that one.

### Phase 4 — Flash attention + softmax + RoPE + RMSNorm

**Goal:** the full per-token forward pass works on wasm32.

- Port `flash_attn_decode` from `runtime-core` (now shared). Use
  `f32x4_relaxed_madd` in the dot + axpy inner loops (already
  prepared in commit 4c09777 via `rayzor_tensor_simd_axpy_f32`).
- Port softmax (scalar OK; not a bottleneck).
- Port RoPE half-split. Pure compute, ~100 LOC.
- Port RMSNorm. Pure compute, ~50 LOC.

**Verification:** synthetic 2-layer transformer (`nue/examples/
tiny-transformer/`) compiles and runs to deterministic output on
both native and wasm32.

**Effort:** 3-5 days.

### Phase 5 — KV cache + sampling + GGUF loader

**Goal:** end-to-end decode is callable from Haxe on wasm32.

- F32 KV cache: port from nue's KVCache.hx. Already pure Haxe, but
  it allocates Tensors — needs Phase 2 done.
- Q8 KV cache: port from `nue-plugins/` (the Q8 KV cache plugin we
  wired this session). Same dequant + fused flash kernel, just the
  WASM build.
- Sampling: top-k + top-p + temperature. Pure compute, scalar OK.
- GGUF loader: pure-Rust parser (mmap unavailable on wasm — use
  `fetch()` + ArrayBuffer in browser; the loader needs an abstract
  byte source). ~500 LOC.

**Verification:** `llama-chat/Main.hx` builds for wasm32. Run via
`wasmtime` and Paris MATCH the canonical prompt. Cool start, no perf
target yet.

**Effort:** 1-2 weeks.

### Phase 6 — Threading (wasi-threads-spec + Web Workers)

**Goal:** multi-core on (a) wasmtime CLI and (b) browser with COOP/COEP.

- `runtime-core/worker_pool`: abstract over `std::thread::spawn` on
  native AND wasm32-wasip1-threads. Same code, both paths.
- Browser path: `runtime-wasm/wasm32_browser_workers.rs` wraps the
  Web Worker spawn via JS imports (`spawn_worker`, `notify`, `wait`).
- Test in `wasmtime --wasi=threads-yes`: parallel_rows works,
  decode tok/s scales.
- Test in Chrome with COOP/COEP server: spawn Web Workers, decode
  tok/s scales (target: 4-6× speedup with 6 workers on M1 Pro).
- Single-threaded fallback: covered automatically because Phase 1-5
  ran single-threaded throughout.

**Verification:** decode tok/s comparison wasmtime-1-thread vs
wasmtime-6-thread; browser Chrome 1 vs 6; demonstrate the gate
(`cache_len >= 256` parallel flash path) kicks in.

**Effort:** 1 week for wasmtime path; 1-2 weeks for browser path
including the JS host module + dev server config.

### Phase 7 — Browser deployment + COOP/COEP affordances

**Goal:** demos work on real browsers without engineers needing to
configure servers.

- `tools/wasm-dev-serve/`: Rust dev server (axum) with COOP/COEP
  headers baked in. Hot reload via WebSocket.
- `examples/wasm-features/llama-chat/`: a browser-side demo. UI
  loads the model file, runs decode, streams tokens.
- `coi-serviceworker` integration documented for GitHub Pages-style
  hosts.
- Single-threaded fallback demo (no headers needed) ships in same
  example.

**Verification:** end-to-end Llama-3.2-1B-Q4_K_M decode in Chrome
with COOP/COEP enabled. Tok/s target: ≥5 (memory-bandwidth-bound,
browser overhead).

**Effort:** 1 week.

### Phase 8 — Perf optimisation pass

Once correctness is end-to-end, port the perf wins from this
session that apply to wasm32:

- Vector FMA peephole in wasm_backend (DONE, commit 4c09777)
- Parallel flash_attn_decode (Phase 6 provides the threading; the
  kernel itself is already in runtime-core)
- thread_local Q8K scratch (Phase 1-3 — works naturally because
  `std::thread_local!` works on wasm32-wasip1-threads)
- USER_INTERACTIVE QoS — N/A on WASM (no QoS API)
- NEON SIMD inner ops — replaced with `simd128` + `relaxed-simd`
  during the per-kernel port

Profile and iterate. Don't pre-optimise.

**Effort:** indefinite — pick wins as they appear.

## Risks & open questions

### R1. WASM relaxed-SIMD support is uneven.

- wasmtime 12+: yes
- Chrome 114+: yes
- Firefox 120+: yes (default-on in Firefox 121)
- Safari 17+: yes
- Node 22+: yes
- Older Node, embedded runtimes: no — module won't load.

**Mitigation:** also offer a build profile with `+simd128 -relaxed-simd`
that emits separate `fmul + fadd` instead of `f32x4.relaxed_madd`.
~3-4% slower; everywhere-compatible.

### R2. wasi-threads-spec is still a proposal.

- Implementation lag: wasmtime is the only mature runtime; wasmer
  added it 2024; wasmedge no.
- Status: it's expected to stabilise in 2026 but no fixed date.

**Mitigation:** runtime-wasm threading is feature-gated; single-thread
fallback is the always-works path.

### R3. SharedArrayBuffer is gated behind COOP/COEP universally.

The Spectre mitigation isn't going away. Any deployment that wants
multi-threaded WASM must commit to the COOP/COEP discipline. This
is a real footprint constraint that affects:

- Embedded analytics (third-party sites): impossible
- CDN-loaded subresources: must be `Cross-Origin-Resource-Policy:
  cross-origin` or `same-origin`
- iframes from cross-origin sources: must `credentialless`

**Mitigation:** single-threaded fallback documented as a
first-class citizen. Demo on rayzor.dev shows the multi-thread
path; embed copy explicitly downgrades to single-thread.

### R4. WASM `i8x16.dot_i8x16_i7x16` (the SDOT analog) is still in
proposal stage.

The native Q4_K_M SDOT path uses ARM/x86 hardware dot products.
WASM has no stable equivalent yet — the relaxed-simd proposal
includes `i8x16.relaxed_dot_i8x16_i7x16_s` but it's not in stable
runtimes.

**Mitigation:** scalar i8 multiply-accumulate inner loop is ~4×
slower than SDOT but still functional. Estimate: 1B-Q4_K_M decode
~8-12 tok/s in single-threaded wasmtime, ~3-5 tok/s in browser
single-thread.

### R5. Memory ceiling.

Llama-3.2-1B Q4_K_M:
- Weights: ~700 MB
- F32 KV cache (max ctx 4096): ~16 MB per layer × 16 = ~256 MB
- Activations: ~50 MB
- **Total: ~1 GB**.

Browser tabs typically allow 2-4 GB per tab. `--max-memory=1073741824`
(1 GiB) in our config is too tight; bump to 2 GiB for Llama-3.2-1B.

**Mitigation:** Q8 KV cache (already in nue-plugins) cuts the KV
share by 3.76× — drops the total to ~840 MB. Land that path on WASM
too.

### R6. GGUF loading without mmap.

Native uses mmap for the GGUF file — zero-copy access to weight
tensors. wasm32 has no mmap; must read the bytes into linear memory.
- wasmtime: `fs::read` works, single allocation
- browser: `fetch()` + `arrayBuffer()` — single allocation, async
- Either way: ~700 MB allocated up-front. Not zero-copy.

**Mitigation:** lazy weight load (read tensor bytes on first
matmul touch) is doable but adds complexity. Defer to Phase 8.

### R7. Build-system gotcha (this session just hit it).

`cargo build --release --bin rayzor` does NOT rebuild
`libnue_plugins.dylib`. Running llama-chat after such a partial
build produces `can't resolve symbol alloc` (JIT path through
`KvCacheQ8.alloc` fails). This is on the WASM side too: any plugin
the WASM target loads needs to be co-built. Recommend:

- Single canonical build command: `make wasm` or
  `tools/build-wasm.sh` that wraps all the steps.
- Add a `rerun-if-changed` build script glue so cargo notices.
- Document it loudly in `CONTRIBUTING.md`.

## Verification approach

- **Per-phase**: smoke tests that exercise the new code paths on
  both `wasmtime` and a headless Chrome (via `chromedriver` or
  `playwright`).
- **Continuous**: a CI job that builds wasm32 + runs `wasmtime`
  smoke after every PR. The native CI keeps running today's tests.
- **Numerical parity**: a `nue/tests/parity/` suite that runs the
  same prompt on native and wasm32; diff token-by-token. Greedy
  decode at temp=0.01 should give bit-identical results; report
  divergence loudly.
- **Perf telemetry**: ship a `tok/s` reporter from the demo, log
  it locally. Don't gate CI on perf — too thermal-sensitive (this
  session learned that lesson hard).

## Effort summary

| Phase | What | Wall (focused engineer) |
|---|---|---|
| 1 | runtime-core extraction | 2-3 days |
| 2 | WASM Tensor + F32 matmul | 3-4 days |
| 3 | Quant kernels (Q8_0/Q4_K_M/Q6_K) | 2-3 weeks |
| 4 | Flash attn + softmax + RoPE + RMSNorm | 3-5 days |
| 5 | KV cache + sampling + GGUF loader | 1-2 weeks |
| 6 | Threading (wasi + Web Workers) | 2-3 weeks |
| 7 | Browser deployment + COOP/COEP | 1 week |
| 8 | Perf optimisation | indefinite |

**Total: 8-12 weeks** for the work-horse engineer to ship Llama-3.2-1B
decode in a Chrome tab with multi-thread Web Workers. Single-thread
working: 4-6 weeks (skip threading until last).

## Decision points / asks

Before committing engineering effort:

1. **Validate the deployment story** with a target user. Is the
   COOP/COEP discipline acceptable for the intended distribution
   model? If "must work embedded in customer iframes without
   their cooperation", we need to plan for single-thread only and
   the perf ceiling drops accordingly.

2. **Pick a v1 model size**. Llama-3.2-1B Q4_K_M (~700 MB) is on
   the heavy side for browser. A Phi-2 / Qwen-0.5B class model
   (~250 MB) is more realistic for the demo. The runtime is the
   same; just the example model file changes.

3. **Code-sharing buy-in**. Option C (runtime-core extraction)
   touches `runtime/` in a structural way. If a separate engineer
   is concurrently changing `runtime/` for other goals, coordinate
   to avoid merge thrash.

4. **CI cost**. Adding wasm32 + Chrome to CI adds ~5-10 minutes
   per PR. Worth it for the test coverage; budget it.

## Out of scope (next plan)

- WebGPU compute backend on WASM. `gpu/src/codegen/wgsl.rs` exists
  for the kernels but the runtime/dispatch layer for browser WebGPU
  needs its own design pass.
- WASI Components (the proposed compositional WASM model). The
  rayzor compiler emits a WASM Component today; whether the
  runtime should split into multiple Components is a separate
  architecture question.
- Different quantisation schemes: AWQ, GPTQ, BitsAndBytes are
  out — Rayzor uses llama.cpp's K-quant family.

## Appendix A — file paths cheat-sheet

| Subsystem | Native crate | WASM crate (today) | After plan |
|---|---|---|---|
| Tensor SIMD | `runtime/src/tensor_simd.rs` | `runtime-wasm/src/lib.rs:2150+` | `runtime-core/src/tensor_simd/` |
| Quant kernels | `runtime/src/quant.rs` | — | `runtime-core/src/quant/` |
| Flash attn | `runtime/src/tensor.rs:2739+` | — | `runtime-core/src/flash_attn.rs` |
| Worker pool | `runtime/src/worker_pool.rs` | — | `runtime-core/src/worker_pool.rs` (abstract) |
| GGUF loader | `runtime/src/gguf.rs` | — | `runtime-core/src/gguf.rs` |
| FFI exports | `runtime/src/plugin_impl.rs` | `runtime-wasm/src/lib.rs` | per-target wrapper |

## Appendix B — COOP/COEP header reference

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

On every cross-origin subresource served alongside the document:

```http
Cross-Origin-Resource-Policy: cross-origin
```

(Or `same-origin` if it's same-origin and the COEP is `require-corp`.)

For credentialed embed:

```http
Cross-Origin-Embedder-Policy: credentialless
```

is a 2024-era looser variant that allows third-party iframes without
their cooperation; lower compatibility — Chrome 96+, Firefox not yet
(as of the doc's writing).

## Appendix C — Reference implementations to study

- **candle** (`huggingface/candle`) — Rust ML inference, has its own
  WASM target. Their `candle-wasm-examples/` shows the wiring for
  browser + Web Worker. Code-sharing pattern: pure-Rust kernels in
  `candle-core/`, multi-target.
- **llama.cpp + Emscripten** — the reference for in-browser LLM
  inference. Their threading uses pthreads via Emscripten's worker
  pool. ~10-15 tok/s for 1B-Q4 on M1 Pro Chrome.
- **wonnx** — ONNX runtime in Rust + WebGPU. Different model
  format but the runtime/deployment patterns transfer.
- **wllama** — llama.cpp WASM wrapper with a worker pool abstraction
  in JS. Good source for the JS-side worker host glue.
