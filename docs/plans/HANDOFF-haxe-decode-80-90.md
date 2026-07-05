# HANDOFF: pure-Haxe llama decode → 80–90 tok/s

Audience: the next agent/session picking this up cold. Everything here is measured from the
current pure-Haxe decode branch after the Tensor.uninit / SpinPool-Int-fix pass. Hardware:
Apple Silicon (M-series).

## Mission & state

Goal: nue llama-chat 1B Q4_K_M decode at **80–90 tok/s (11–12.5 ms/token)** on the pure-Haxe
kernel path (`RAYZOR_HAXE_MATMUL=1`), no Rust FFI matmuls.

| metric | 2 days ago | NOW |
|---|---|---|
| decode step p50 | 164 ms | **21.5–25.75 ms** (thermal band), user-sustained ~35–40 tok/s |
| prefill | 157 ms/token | **~20 ms/token** (0.33 s / 17-token prompt) |
| Rust-FFI reference, same binary | — | 9.5 ms/step (105 tok/s short-run; 85–90 sustained) |

All kernel changes are bit-exact by construction (per-row reduction order preserved);
`test_q4km_dot` is the canary.

Latest landed pass:
- `Tensor.uninit(shape, dtype)` exists for full-overwrite producers. Native runtime skips the
  zero-fill on pool hits and fresh allocations; wasm/JS fallbacks keep zero-init for safety.
- `Q4Matmul` uses `Tensor.uninit` only for outputs it fully overwrites.
- Pure-Haxe `Linear.forward` no longer clones the input before `Q4Matmul.matmul`.
- `SpinPool.parallelRows` keeps the original worker-state layout/protocol; only `Int`
  annotations were added to prevent the `ushr.f64` trap-stub regression.
- Focused validation: release build green; `test_tensor_uninit_full_overwrite`,
  `test_q4km_qmatmul`, and a 64-token llama smoke pass with no `W0020` / no LLVM tier loss.

## How to run / validate

```bash
# build (ALWAYS after compiler/runtime/haxe-std edits; nue/*.hx needs no build)
LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm CARGO_INCREMENTAL=0 cargo build --release -p rayzor --bin rayzor

# model run (MUST: --no-cache after editing any imported .hx; pkill before/after)
cd nue/examples/llama-chat && pkill -9 -f target/release/rayzor
RAYZOR_PROFILE_DECODE=1 RAYZOR_HAXE_MATMUL=1 ../../../target/release/rayzor run Main.hx \
  --no-cache --safety-warnings off -- <GGUF> "prompt" 32
# read: step_p50_ms (decode), prefill_fwd_s / prompt_tokens (prefill)

# suite (must stay green; run after every compiler change)
./run_haxe_tests.sh     # 158/158 at handoff
```

Kernel micro-bench harness (single-thread, resident vs streaming): `/tmp/stream_bench.hx`
pattern — self-contained copy of the kernel; **MUST run with `--release`** (see gotcha #1).

## Architecture (the 60-second tour)

- `nue/nue/Q4Matmul.hx` — the kernel. `matmul` (single weight), `matmulFused` (up to 3 same-K
  weights: ONE Q8_K quantize + one dispatch over concatenated rows), `runBanded` (banded pass
  against pre-quantized scratch), `quantizeAll`/`quantizeBlock` (block-linear packed scratch:
  elements at g*256, bsums at g*64, scale dOut[g]; banded when ≥16 blocks). Per-block dots:
  `q4DotMA4`/`q6DotMA4` (SDOT via `SIMD4i32.dot`; ADDV via `.sum()`; exact f16 header decode
  via `Mem.f32FromBits`). IR-verified fully inlined: zero extern calls in the hot loop.
- `compiler/haxe-std/rayzor/Mem.hx` — address-based single-instruction load/store intrinsics
  (`loadF32/storeF32` = the ONLY scalar f32 access from Haxe; `f32FromBits`; `prefetch`).
  `Bytes.loadI32AlignedUnchecked` pair = handle-based equivalents.
- `compiler/haxe-std/rayzor/concurrent/SpinPool.hx` — persistent chunk-STEALING pool
  (atomic-cursor claims, bit-identical row ownership). Idle policy is self-tuning:
  tight-spin (2k) → yield-hold (`Thread.yieldNow`) sized by 4× the pool's inter-dispatch-gap
  EWMA → park (`Parker`). 7.4× on 8 claimants standalone.
- Core partition: `runtime/src/worker_pool.rs` defaults the RUST pool to **1 worker under
  RAYZOR_HAXE_MATMUL** (its spin-wait workers otherwise war with the guest pool: decode was
  65 ms/token before this, 27.5 after). `RAYZOR_WORKERS` overrides.
- Tiering: default `rayzor run` auto-installs the LLVM tier before main (Application preset).
  Everything perf-relevant runs LLVM; Cranelift is startup/fallback.
- llama-chat glue: GQAttention QKV + SwiGLU gate/up go through `matmulFused`. Remaining Rust
  FFI per token ≈ 1.5 ms total, all µs-scale kernels: silu, flash_attn_decode (seqQ==1 only —
  prefill attention takes the unfused bmm path), rope, rms_norm, reshape (520 calls),
  free/clone, softmax, add_into, topk_scan, + Bytes.alloc per matmul.

## Decode budget @ ~23 ms (where the remaining 2× lives)

- matmul band ≈ 17 ms = ~102 ms serial work at ~6× parallel efficiency (10 claimants)
- FFI glue ≈ 1.5 ms; quantize banded; sampler/decode ≈ 0.3 ms
- Per-thread rate in-model ≈ 11.2 GMAC/s vs 13.8 resident-bench; Rust ≈ 14.8 in-model.

### Live-decode sample findings (2026-07-06, `sample <pid>` during 512-token decode)
- Band lambdas: 2.8k leaf samples spread FLAT across offsets — no single stall point in the
  kernel; the per-thread-rate residue is diffuse (micro-arch: needs Instruments-level IPC/
  stall attribution, not more code reading).
- System is wait-dominated: psynch_cvwait 25k + swtch (yield) 13k — workers idle ~92% because
  band work is only ~1.7 ms/thread/token. Parallel-efficiency headroom is in the JOIN tail and
  the non-band ~6 ms (FFI 1.5, attention on the 2-thread Rust pool, tensor mgmt, sampler).
- FIXED from the sample: SpinPool.cell() re-derived the ctl base via extern address() per
  atomic access (hot in all spin loops) — Atomic handles now hoisted per loop.

### Ranked next steps
1. **Per-thread rate 11.2 → ~13.8**: flat profile ⇒ diffuse; use Instruments (or PMU counters)
   for IPC/stall attribution on the band thread. 2-row Q6 tiling (+17.7% on the Rust side)
   remains the one untested structural idea — shared operand across 2 rows is activation+bsums;
   multi-return is the blocker (no out-params without heap; consider accumulating row-pair
   results via two independent inline chains in the band body itself).
2. **Trims still open**: 2 Bytes allocs per matmul (`qs`, `bsums`) + `Array<Float>` dScales
   remain hot. Next safe trim is persistent per-model/per-pool scratch (instance-held; NEVER a
   cross-module static, see gotcha #3). `Tensor.uninit` is already landed; do not redo it.
3. Memory-leak backstop (separate from perf): InsertFreePass has no MakeClosure arm and can't
   see extern-returned containers (shape() arrays, fused triples); band-closure env+struct leak
   per matmul (~small); GB-scale suspect if RSS climbs again: flash-gate miss → unfused bmm
   chain (~150 MB/token) — add a one-shot counter at GQAttention's gate-miss branches.

### Session results 2026-07-06 (late)
- LANDED 90323ba7: u32 header decode (kmask shape) + two-block paired inner loop → sustained
  **40-41.7 tok/s** (256 tok, rested), p50 21.5-22 ms. cd8de0c8: Atomic handles hoisted out of
  SpinPool spin loops (cell()/address() extern was sampled hot).
- RAYZOR_WORKERS sweep {1,3,6} on sustained runs: **1 remains best even with yield-hold**
  (w=6: p95 66 ms — core war returns). Long-context droop (35.3 @512 vs 40.6 @256) is partly
  attention-on-1-worker growth; partly thermal.
- THE REMAINING 2×, quantified: FFI aggregate ≈160 GMAC/s ⇒ Rust **~21.7 GMAC/s per thread
  in-model**; our isolated best is 13.7. The phase-3 "1.08× parity" compared a weaker Rust
  entry. The 1.6× kernel gap survived: header opt (wash), pairing (+4%), per-p partial
  restructure (wash, reverted). NEXT: instruction-level comparison — disassemble our inlined
  block loop (fn-ptr dump + lldb) vs Rust dot_q4_k_q8_kblock_2 objdump; count instructions and
  SDOT density per block. That tells us if it's codegen quality (fixable in the LLVM pipeline /
  MIR shapes) or an intrinsic-level difference.
- **TOOLING TRAP: test_q4km_qmatmul is SELF-CONTAINED — it does NOT exercise nue/Q4Matmul.**
  A deliberately wrong imin mapping passed it with an identical error value. Any kernel edit
  must be validated with a model run; write a real canary that imports nue.Q4Matmul.

## FLASH-DECODE HAXE PORT (the long-context fix — darmie directive, spec'd 2026-07-06)
WHY: attention is the LAST GROWING FFI term in decode. It is SERIAL in the current kernel
(RAYZOR_WORKERS 1/2/3 identical at 800 tokens = 32.4 tok/s) and scales with context: 41.7
tok/s @256 ctx → 35 @804. By ctx 800 it is ~5-6 ms/token and climbing. Port it to the guest
pool, banded over the 32 q-heads — parallelism IMPROVES with context.

Reference kernel (mirror exactly): runtime-core/src/tensor/flash_attn.rs
`flash_attn_decode_one_qhead` — per q-head, 3 passes over cache_len:
 (1) scores[l] = dot(q_head[64], K[l, kv_head(=q_head/group), :64]) * scale; track max;
 (2) softmax: scores[l] = exp(scores[l]-max), denom += (max-shifted, matches Tensor.softmax
     reduction order); (3) out[64] = Σ scores[l]/denom * V[l, kv_head, :64].
Decode path uses the Q8 cache: GQAttention.hx:187 → KvCacheQ8.flashAttnDecodeQ8Host/Q8
(nue/nue/transformer/KvCacheQ8.hx — extern; NO raw-pointer surface yet).

Port plan:
1. Runtime getters (trivial): KvCacheQ8.dataPtr():Usize + rowStrideBytes():Int + the Q8_0
   block layout constants (32 quants + scale; confirm f16 vs f32 scale in runtime/src source
   of kv_cache_q8 before writing the dot).
2. Kernel Q4Matmul-style in a new nue/nue/FlashDecode.hx: band over q-heads via the SpinPool
   (32 bands; per-claimant scores scratch = ONE Bytes.alloc(32*ctx*4) per call before
   dispatch, slice by head — NO statics, see the materialized-copies gotcha). Scores pass via
   SDOT: quantize the 32 q-head rows (64 f32 each) to Q8 once per call (reuse quantizeBlock
   pattern, per-head scale), then q8xq8 SDOT against K blocks (2 SDOT per 64-dim row) — the
   wasm q8 flash kernel (perf_wasm_q8_flash_simd128 memory) is prior art. V pass: f32 axpy
   via SIMD4f mul/add over dequantized V rows (dequant inline per 32-block: SIMD16i8.load +
   ... no i8->f32 widen op exists — scalar dequant of V may suffice: V pass is
   bandwidth-light vs scores; measure before adding a widen primitive).
3. Numerics: exp = Math.exp (LLVM intrinsic); max-shifted softmax reduction order must match
   Tensor.softmax exactly; compare logits against the FFI path on a fixed prompt for N steps
   (NOT the vacuous canary pattern — validate through the model).
4. Wire: GQAttention decode branch behind Linear.useHaxeMatmul(), FFI fallback kept.
Expected: removes the ctx-linear FFI term; flat ~40+ tok/s at long context; combined with the
kernel-rate disasm work = the 80-90 path.

## TIERED-RUNTIME REDESIGN (agreed direction 2026-07-06, unstarted — TOP priority next session)
Cold start measured: before-main LLVM upgrade costs **+3.0s to first token** (5.97 vs 2.96s).
The ladder is currently two-tier with dead middle (Baseline/Standard/Optimized = identical
cranelift-"speed" codegen; warm/hot thresholds in llama-chat manifest are inverted 1/30/5 and
unvalidated; count-based promotion can't see inner functions; mid-run swaps are no-ops because
calls bake absolute relocations).

The keystone: **patchable call table**.
1. TieredBackend allocates a leaked slot table (dense slot per function; CraneliftBackend
   assigns slots at declare, exports func_id→slot). Baseline-tier CallDirect lowers to
   iconst(slot_addr) → load → call_indirect (sig via import_signature from called_func.
   signature). Touch point: cranelift_backend.rs ~3326 (normal-call func_ref) and the call
   emission below it (~3340-3500: sret/env/arg handling stays identical; only the callee
   operand changes). Extern/libc/runtime calls stay direct. Fill slots at finalize; LLVM
   upgrade atomically stores its pointers into the SAME slots → promotion lands mid-run
   (fat-ptr thunks are cranelift bodies whose calls route through the table, so existing
   closures/vtables promote transitively). Flag-gate (config/env) until proven.
2. Background LLVM: spawn the whole-module compile on its own thread (own Context) at main
   entry; swap table slots when done. Cold start = cranelift (2.96s), full LLVM speed lands
   seconds later mid-run. (MCJIT bulk-resolution history: resolve per-function; keep the
   sync-before-main path as fallback flag.)
3. Warm tier that carries weight: baseline compiles at cranelift "none" (faster cold start);
   per-function hotness counters injected in baseline prologues (same injection point as the
   shadow-stack instrumentation) route hot functions through the EXISTING beadie background
   per-function recompile at "speed", swapped via the table (ms-scale per function). Hot/max =
   the background LLVM whole-module swap. Validate thresholds (warn on inverted configs).

## AUTOVECTORIZATION: answered 2026-07-06 (probe /tmp/vec_probe.hx)
Loops ARE JIT'd both tiers, but: cranelift never autovectorizes; on the LLVM tier loop/SLP
vectorizers are ON yet **FP reductions cannot legally vectorize because our fast-math runs
WITHOUT reassoc** (deliberate, bit-exactness) — a clean raw-f32 sum loop compiled fully scalar
(no vector ops in post-opt IR; 4.2 GB/s = fadd latency chain). Integer loops may vectorize;
float accumulations never will. Hot kernels are fast because of EXPLICIT SIMD — that is the
pattern for any new hot loop. Optional lever: per-function @:fastMath metadata → set reassoc
on that function's FP ops (bit-exactness preserved elsewhere), or manual multi-accumulator
loops.

### Measured NEGATIVE results — do NOT redo
- **Software prefetch in the 1B decode kernel**: +8%/token REGRESSION (reverted). The 1B active
  weight set (~32 MB/token) is LLC-resident across tokens; prefetches are pure overhead.
  Infrastructure kept (`Mem.prefetch` → llvm.prefetch, no-op elsewhere): single-thread
  STREAMING bench = 8.5 → 13.5 GMAC/s (98% of resident) — it IS the lever for 8B-class models.
- **Long static spin budgets**: 2M spins measured 23.75 cold but 184–255 ms heat-soaked
  (starves coexisting native pools). The adaptive policy replaced this; don't reintroduce.
- **Haxe SwiGLU elementwise**: serial Haxe loses to the two µs-scale FFI kernels it replaces.
- **RAYZOR_WORKERS=0**: worse (some FFI kernels still need a thread).
- Q6 restructure to integer-domain: the native Rust reference (runtime-core/src/quant/sdot.rs
  `dot_q6_k_q8`, q6_k.rs simd128) is structurally IDENTICAL to our port — no free win there.
- **SpinPool claimant-count cap / worker-state layout change**: REGRESSION. Waking fewer
  arbitrary worker slots is not a P-core selector; `RAYZOR_HAXE_POOL_CLAIMANTS=8` measured
  ~109.75 ms p50 and the control-cell layout change caused cache-sensitive failures. Keep the
  original worker-state layout. P-core-only needs topology/affinity at worker construction, not
  dispatch-time claimant caps.
- **Per-p partial-reduction restructure of q4DotMA4** (Rust dot_q4_k_q8 shape: fold each
  sub-block pair into scalar partials immediately): WASH vs the committed 8-acc + single-hsum
  shape (extra hsums cancel the register-pressure relief). Reverted.
- **Q4Matmul branch-hoist of `isQ6 ? q6Dot : q4Dot` out of the inner loop**: REGRESSION in the
  model path (~35.75 ms p50 in a profiled smoke) despite preserving numerics. The current backend
  likes the original ternary shape better; do not redo without inspecting generated LLVM.
- **Persistent scratch via SAME-FILE private statics in Q4Matmul**: SIGSEGV. matmul/matmulFused
  are materialized into caller modules (Linear/SwiGLU are separate files) and those copies see
  duplicated/garbage object statics — the static hazard applies even within one package. The
  only safe home for persistent scratch is instance plumbing (like Linear.pool); worth ~0.3
  ms/token, do it only if adding a Linear field proves drift-safe.
- **Haxe-side shutdown pool profiling** (`"[profile-pool] " + spinPool.profReport()` in
  `LlamaModel.shutdownPool`) caused LLVM verifier failure (`Invalid bitcast i32 -> i64`) and
  silently dropped the program to Cranelift. If pool timing is needed, prefer a runtime-side
  counter dump or a compiler fix first.

## GOTCHAS — these WILL bite you (each cost us hours)

1. **Benching without `--release` is meaningless**: the Application preset injects shadow-stack
   frame calls into every function (`rayzor_push/pop_call_frame`) — a hot kernel runs ~19×
   slower (0.7 vs 13.7 GMAC/s). llama-chat's manifest sets `enable_stack_traces=false`, so
   MODEL runs are exempt; standalone benches are NOT.
2. **Silent tier loss**: ANY LLVM verifier failure silently drops the WHOLE program to
   Cranelift (grep stderr for "upgrade failed"). Three instances fixed (CallIndirect int
   widths, float↔int backstop, stealLoop's Int-local decaying to Float under the
   `if (a > b) a = b` reassign shape — annotate `:Int` in doubt). If perf suddenly ×0.1,
   check this FIRST.
3. **Cross-module member drift**: adding methods/fields to a class used across modules
   (SpinPool!) can mis-dispatch importers' calls to OTHER methods (`spinPool.shutdown()` from
   LlamaModel currently no-ops — workers are abandoned at exit; known). Statics don't forward
   across modules at all (instance-plumb everything, e.g. `Linear.pool`). General fix =
   bugs_import_xmodule_member_resolution (open).
4. **`Sys.*` calls in constructors or hot dispatch paths trap or mis-resolve** (exit 132/133).
   The ONE proven-safe place for env reads near the pool is the lazy-init block at the top of
   `SpinPool.parallelRows`. Model with care.
5. **`Null<C>`-typed receivers**: method calls used to be SILENTLY DROPPED (fixed afeb76f8 —
   `resolve_type_to_class_symbol` Optional arm). The general disease — unresolved method call
   lowering to NOTHING instead of a hard error — is still open: TAST emits the MethodCall with
   a placeholder symbol; something in tast_to_hir/hir_to_mir eats it. Add the E08xx hard error
   (no-silent-fallthrough is a standing project rule).
6. **`--no-cache` after editing any imported `.hx`** or you run stale MIR (symptom: exit 132
   traps that look like brand-new bugs). **pkill rayzor before/after runs** (orphans spin a
   core and invert A/Bs). **Heat-soak inverts A/Bs ~3×** — alternate configs within the same
   thermal window, rest the machine for absolute numbers.
7. Method-name mismatches compile to trap stubs, silently until called (`Thread.yieldNow`,
   not `yield_now`). Exit 133 + no output = look for an unresolved extern.
8. `--tier-promotion false` (CLI) with llama-chat's manifest routes main through the
   interpreter and breaks; change promotion in the `[tier]` block instead.
9. `Tensor.uninit` is intentionally unsafe: only use it where every element is written before
   any read. Native skips zero-fill; wasm/JS fallback currently aliases it to zero-init.

## Key files
- Kernel: `nue/nue/Q4Matmul.hx`; glue: `nue/nue/transformer/{GQAttention,SwiGLU}.hx`,
  `nue/nue/Linear.hx`, `nue/nue/arch/{LlamaArch,LlamaModel}.hx`
- Pool: `compiler/haxe-std/rayzor/concurrent/{SpinPool,Parker}.hx`; Rust pool:
  `runtime/src/worker_pool.rs`
- Intrinsics: `compiler/haxe-std/rayzor/Mem.hx`, `compiler/src/stdlib/bytes.rs`,
  mapping rows in `compiler/src/stdlib/runtime_mapping.rs`, LLVM extern-replacement chain in
  `compiler/src/codegen/llvm_jit_backend.rs` (try_create_*_intrinsic)
- Tier config: `nue/examples/llama-chat/rayzor.toml` `[tier]`
- History/attribution detail: memory file `perf_haxe_kernel_pool_cooperation_blocker.md`
  (path: `~/.claude/projects/-Users-amaterasu-Vibranium-rayzor/memory/`)
