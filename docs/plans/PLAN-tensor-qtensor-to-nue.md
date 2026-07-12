# PLAN: Move `Tensor`/`QTensor` out of Rayzor core into Nue (pure-Haxe + platform backends)

Status: PLANNED (pick up later). Author handoff for Claude/any next agent.
Prereqs in flight: env-var decoupling (keep Rayzor core env-pure), pool-profile
work already landed (`1fab8e78`).

## 0. Confirmed architecture (3 layers)

- **Rayzor core** — general language/compiler/runtime, PURE. General primitives
  (`Bytes`, `Atomic`, `Ptr`, `Mem`, `String`, `Array`/`Map`, concurrency,
  `SIMD4f`) + compiler intrinsics (the only things that inline/fuse). NO ML, NO
  GPU, no inference env, no downstream doc-context.
- **Nue** — the ML engine. `Tensor`/`QTensor` (`nue.*`), pure-Haxe portable CPU
  kernels as the **baseline that runs on every target (native + wasm)**.
- **Plugins (FFI-bound, opt-in accelerators)** — platform-specific backends
  behind the Nue backend interface; pure-Haxe is always the fallback:
  - Apple Accelerate/AMX (CPU) — plugin.
  - **GPU / Metal / MPS — a plugin. GPU is inherently FFI** (needs the native
    GPU API); it is NOT core and NOT pure-Haxe. Today's `compiler/tests/gpu/*`
    Tensor-on-GPU tests move with Tensor into the Nue/GPU-plugin world.
  - x86 VNNI — intrinsic or plugin.

Decision (user, 2026-07): **full move** — both `Tensor` and `QTensor` leave
`rayzor.ds` for `nue.*`; Rayzor stdlib keeps NO tensor types; GPU is a plugin.
The 11 Rayzor `test_tensor_*`/`test_gpu_*` tests relocate to Nue (or drop).

## 1. Vision / end state

- `Tensor` and `QTensor` become **Nue** types (`nue.*`), not Rayzor stdlib
  (`rayzor.ds.*`). They are ML/inference types; Rayzor core stays general.
- Nue owns a **pluggable kernel backend** behind one interface:
  - **PureHaxe** — portable default (`Mem`/`SIMD4f`/`Bytes` intrinsics). Runs on
    every Rayzor target (native AND wasm) with no per-platform FFI.
  - **AppleAccelerate** (macOS) — FFI to Accelerate BLAS/BNNS (AMX under the
    hood). See `project_untapped_compute_levers_apple.md`.
  - **x86VNNI** — the existing `vpdpbusd` path for AVX-VNNI (NUC/Alder Lake).
  - Selected at runtime by platform detection + `NUE_TENSOR_BACKEND` override.
  - Pure-Haxe is the **bit-exact reference**; accelerated backends validate
    against it (exact for int paths, tolerance for AMX/fp).
- **Delete ~8.5k lines** of inference from `rayzor-runtime` (`quant.rs` 3.5k +
  the ML half of `tensor.rs` 5k) once the Haxe port is validated. Core becomes
  genuinely general-purpose; every inference env var leaves core.

Why: NOT the ~2.4% residual FFI perf (small). The prizes are (a) a pure/general
Rayzor core with zero inference tampering surface, (b) **cross-platform for
free** (one Haxe impl for native + wasm; today the wasm path needs its own
kernel copies), (c) "ML lives under `nue.*`" finally consistent, (d) a clean
place to add Apple/x86 accelerated backends without polluting core.

## 2. Current state (2026-07 — what's already pure-Haxe vs FFI)

Already pure-Haxe (the hard parts — done): `matmul` (`Q4Matmul.hx`), `flash`
(`FlashDecode.hx`), `quantize`, `silu` (`SwiGLU.hx`), `softmax`
(`KvCacheQ8.hx`). Production matmul (`NUE_MATMUL=1`) does **0** FFI calls in the
hot loop (`Linear.hx:112` → `Q4Matmul.matmul`; 48 `Mem`/`SIMD4f` intrinsic
uses, 0 `qtensor_`/`rayzor_tensor_`).

Still FFI-backed (measured, 200-tok decode @ ~86 tok/s, `RAYZOR_KERNEL_TIMING`):
total FFI ≈ **0.28 ms/token ≈ 2.4% of decode**. Breakdown:
- `tensor_rope`   30.4%  (~32 calls/tok, 2.68 µs) — `RoPE.hx` delegates to FFI.
- `tensor_rms_norm` 27.2% (~33 calls/tok, 2.32 µs) — `RMSNorm.hx` delegates.
- `tensor_free`   15.3%  (**~374 calls/tok**, 0.115 µs) — temporary churn.
- `tensor_reshape` 10.5% (~65 calls/tok) — metadata / temporary churn.
- `topk_scan` 8.1%, `tensor_add_into` 6.0%, `tensor_softmax` 2.3%
  (prefill-ish), `tensor_clone` 0.3%.

Data structures: `Tensor`/`QTensor` are **Rust handles** (buffer + shape +
refcount in `tensor.rs`/`quant.rs`), bound as **Rayzor stdlib** externs
(`compiler/haxe-std/rayzor/ds/Tensor.hx`, `QTensor.hx`; `@:native("tensor_*")`,
`@:native("qtensor_*")`). No intrinsic/inline privilege — every FFI op is a
materialized-address indirect call, identical to a plugin native (verified:
`llvm_jit_backend.rs:603` `inttoptr`, `compilation.rs:7682` plugin symbols).
So moving the kernels to the Nue plugin is call-mechanism-neutral; true
inlining only ever comes from **compiler intrinsics** (`Atomic`/`Mem`/`SIMD`),
which stay in core and are what the pure-Haxe kernels are built on.

## 3. Phases

### Phase A — Finish the op-math ports (low risk, shrinks FFI to ~0)
Port the remaining elementwise/reduction loops from Rust to Haxe `Mem`/`SIMD4f`,
same playbook as matmul. Order by FFI share:
1. `rope` — rewrite `RoPE.apply` to rotate Q/K in place over the buffer with
   precomputed cos/sin (no fresh-tensor alloc — kills the "rope allocates fresh
   output tensors" churn too).
2. `rms_norm` — sum-of-squares (SIMD hsum) + scale, in place.
3. `add_into` (residual), `reshape` (metadata/view — see Phase B), `clone`.
4. `topk_scan` is already NEON-optimized on the FFI side; port last / keep.
Validate each bit-exact vs the Rust op before switching the default (keep Rust
reachable behind `NUE_TENSOR_BACKEND=ffi` during the transition).

### Phase B — Pure-Haxe `Tensor`/`QTensor` data structure (the real design work)
- Haxe `Tensor` = raw `Bytes` buffer + shape (Int array or packed) + dtype +
  lifecycle. **No `Array<Float>`** anywhere (per-element boxing bug — §5). All
  access via `Mem.loadF32/storeF32` / `Bytes` unchecked accessors.
- **Views / in-place / pooled buffers** to erase the alloc churn: `reshape` and
  slices become views (share buffer, new shape), not new allocations. Target:
  drop the ~374 `free` + 65 `reshape` + `clone` per token (the ~26% of FFI that
  is pure waste, not compute).
- Lifecycle: reuse the `SpinPool.scratchBytes` pooling idea for transient
  activations; explicit free or arena per forward pass. Avoid GC pressure from
  large f32 buffers.
- `QTensor`: quantized block storage (Q4_K_M/Q6_K) as `Bytes`; `from_bytes`,
  `dequant`, `requant`, `gather_rows` ported (load-time, low perf priority but
  needed to delete `quant.rs`). GGUF parsing partly in `GGUFLoader.hx` already.

### Phase C — Relocate + delete core

Mechanics (verified): `haxe-std/` is UNIVERSAL classpath (every program sees
`rayzor.ds.*`); `nue/` is added per-program via `rayzor.toml` `class-paths`
(e.g. the llama-chat example: `class-paths=[".","../../"]` reaches `nue/`).
`rayzor run` has NO `-cp` flag — classpath comes from `rayzor.toml`. So:
- The 5 `rayzor.ds` files (`Tensor`, `QTensor`, `DType`, `Device`, `QScheme`)
  are INTERDEPENDENT (`Tensor`→`DType`/`Device`, `QTensor`→`QScheme`) → the
  relocation is ATOMIC, no canary.
- The 11 Rayzor tests (`compiler/tests/haxe/test_tensor_*`, `test_gpu_*`,
  `test_q4km_qmatmul`) lose `rayzor.ds.*` → re-home to a `nue/tests/` dir with a
  `rayzor.toml` `class-paths` reaching `nue/` (preserve coverage), or drop.
- RISK: `Tensor`/`QTensor` are the most-imported types (42 files). This move is
  a live trigger for the x-module member-resolution cluster (§5 #10/#11:
  static-not-forwarded, instance-field SymbolId drift, same-name collapse) —
  subtle miscompiles that surface only as WRONG MODEL OUTPUT. MUST validate
  bit-exact argmax vs the FFI baseline on the long Voronoi prompt after the
  move, not just "it compiles/runs".

Execution order (each a validated commit):
1. Copy 5 files `rayzor.ds` -> `nue/nue/ds/`, `package nue.ds`; sed
   `rayzor.ds.*` -> `nue.ds.*` in `nue/` + `tools/` + `examples/`. Delete the
   old `rayzor.ds` files. Build + run model, **argmax-match vs baseline**.
2. Re-home the 11 tests (new `nue/tests/` + `rayzor.toml`), run them.
3. (later) kernels `quant.rs`/`tensor.rs` ML half -> `nue-plugins`.
- Move `Tensor.hx`/`QTensor.hx` from `compiler/haxe-std/rayzor/ds/` → `nue.*`
  (e.g. `nue/nue/ds/`). Update all nue imports. General stdlib externs
  (`Bytes`/`Atomic`/`Ptr`/`Mem`/`String`/concurrency/`SIMD4f`) are untouched —
  they don't depend on tensor kernels, so the "don't break stdlib externs"
  constraint holds.
- Any kernels kept as FFI (accelerated backends) live in the `nue-plugins`
  crate (loaded via `--native-lib`, `RTLD_GLOBAL`; already exports 12 natives).
- Delete `quant.rs` + the ML half of `tensor.rs` from `rayzor-runtime` **only
  after** every op is validated bit-exact. Keep them (or a slimmed copy) as a
  test-only oracle if useful.

### Phase D — Platform-specific accelerated backends
- Backend interface on `nue.Tensor`/`QTensor`: `matmul`, `flash`, `rope`,
  `rms_norm`, etc. PureHaxe implements all (portable baseline).
- **AppleAccelerate** (macOS, `#if macos`): FFI to `cblas_sgemm` / BNNS for the
  F32/dequantized matmul path (AMX). Big-kernel indirect-call overhead is
  negligible; AMX throughput is the win. Quantized path may still be pure-Haxe
  (Accelerate has no Q4_K_M). Validate vs pure-Haxe within fp tolerance.
- **x86VNNI**: keep the `vpdpbusd` VectorDot path for the NUC.
- Runtime selection: platform default + `NUE_TENSOR_BACKEND=haxe|accelerate|vnni|ffi`.
  Pure-Haxe is always the fallback (and the wasm path).
- Philosophy note (from HANDOFF): pure-Haxe is the goal/baseline; accelerated
  FFI backends are **opt-in accelerators**, not a fallback-to-Rust. They must
  never silently replace a validated pure-Haxe result.

## 4. Validation harness (mandatory, do first)
- Per-op bit-exact test: pure-Haxe op vs the Rust reference on fixed inputs,
  temp-0. int paths bit-exact; fp within a documented ULP tolerance.
- A `chrepro`-style matrix of tensor shapes/dtypes (like the channel matrix).
- Full-model coherence: argmax match vs the FFI baseline on the long Voronoi
  prompt at temp 0 (the project's standard coherence check).
- ALWAYS verify codegen by disasm for the SIMD ops (SDOT/hsum) — never trust
  IR/source (see `feedback_verify_codegen_by_disasm`, the "SDOT never fired
  in-model" saga: MCJIT generic subtarget + i8mm instcombine).

## 5. Compiler-bug catalog to navigate (the real friction, ~30+)

The Nue port already hit these. A full `Tensor` port will re-encounter them.
Each is workaround-able; the discipline column is the rule. Grep the memory dir
for the named files for repro + detail.

### A. Numeric / boxing (the kernel killers)
1. **float-conditional-reassign boxes** (`bugs_float_conditional_reassign_boxes`)
   — `if(c) f=expr` + loop-carried `Float` + `Math.floor/ceil` BOXES per element
   → multi-GB leak. → ternary→`fcsel`, `Std.int`→`fptosi`; general fix OPEN.
2. **Array<Float> per-element boxing** (`project_haxe_kernel_audit_2026_07_05`,
   `681086a5`) — `Array<Float>` element access on native = one heap alloc per
   read (60M/tok, +800MB). → **never `Array<Float>` in kernels; raw `Bytes` +
   `Mem`.**
3. **inferred-int bitwise/float-div** (`bugs_inferred_int_bitwise_float_div`) —
   `(bits & mask)` int-truncates a float. → annotate `:Int`.
4. **Int locals decay to Float in MIR** under the `if-reassign-min` shape
   (`stealLoop` needed explicit `:Int`) — verifier fail → silent Cranelift drop.
5. **Bytes accessors opaque externs on native** (`llvm_jit_backend.rs:656`,
   audit #1) — 77M `getInt32`/tok; use `Mem.load*` address-based accessors.

### B. Null / coercion / fields
6. **Null<Int>→Int cross-file garbage** (`bugs_null_int_cross_file_assign`) →
   `+0` workaround.
7. **null primitive field GEP corruption** (`bugs_null_primitive_field_gep_corruption`)
   — class field `Null<Int/Float/Bool>` after a scalar field gets wrong GEP
   elem_size (1 not 8) → SIGSEGV.
8. **StringMap.get(missing)→raw 0** (`bugs_stringmap_null_get`) → use `exists()`.
9. **Null<C> receiver method call silently drops to nothing**
   (`afeb76f8`, resolve_type_to_class_symbol had no `Optional` arm) — the call
   lowered to a line-marker; pooled kernel wrote zeros. OPEN: needs a hard
   E08xx error, not silent fallthrough.

### C. Cross-module resolution (the "ordering disease")
10. **x-module resolution cluster** (`bugs_xmodule_resolution_disease_cluster`)
    — 5 shapes: extern-return Float decay; cond-int-reassign incl f16;
    same-package import; extern-method stubs; same-name cross-class collapse;
    `enabled()`-in-ctor. Worked-around; root fix = TAST declaration-based
    resolution + hard-fail.
11. **x-module member resolution** (`bugs_import_xmodule_member_resolution`) —
    order-dependent; instance fields (SymbolId drift) + statics never forwarded.
12. **x-module Dynamic/reflection** (`bugs_cross_module_dynamic_object_reflection`)
    — class→Dynamic arrives void; `isOfType` wrong; `Reflect` crashes;
    String→Dynamic UNSAFE.
13. **dtype enum → wrong x-file enum** (`bugs_dtype_enum_cross_file_pointer`,
    shadowing) — bare enum resolves to the wrong module's enum. type-driven
    FIXED; bare OPEN.
14. **cross-module lambda name collapse** (`bugs_native_channel_spawned_receiver`,
    fixed `767806a0`) — `<lambda_0>` per file collapsed by name; backend binds
    by name. Fixed by module-qualified lambda names.
15. **adding methods to an x-module class breaks importers' dispatch of OTHER
    methods** (member-layout drift) — kept eating pool instrumentation.

### D. Generics / collections / patterns
16. **generic monomorph untyped ctor** (`bugs_generic_monomorph_untyped_ctor`)
    — `new Foo<Int,Int>()` + x-module chains fixed; residual `List.iterator`/
    `Map.keys` for-in panic.
17. **Map for-in iterates keys** (`bugs_map_for_in_iterates_keys`) — `for(v in
    map)` fixed; `m.keys()` still panics.
18. **switch-expr ctor-pattern lowering** (`feedback_switch_expr_phi`) — prefer
    if/else over switch-expr returning a value in hot/ctor paths.
19. **enum mixed-payload map corruption** (`bugs_enum_mixed_payload_map_corruption`).
20. **RD parser** (`bugs_rd_parser_issues`) — `Array<T>=expr` lexes `>=`.

### E. Codegen / tier / SIMD
21. **Silent tier-loss** — any LLVM verifier failure silently drops the WHOLE
    program to Cranelift (CallIndirect int widths `17bdedc2`; float↔int backstop
    `c33c25ab`). Now errors instead of null-lowering (`llvm_jit_backend`).
22. **SDOT never emitted in-model at LLVM** (`456521f0`) — MCJIT empty MCPU →
    generic armv8.0 ISel; O3 instcombine `sext(and)`→`zext`→USDOT→needs i8mm
    (absent on M1). Fix: emit `llvm.aarch64.neon.sdot` DIRECTLY. **Verify by
    disasm.**
23. **SIMD4f** (`bugs_simd4f_broken_two_part`) — 5 bugs fixed; `.set(lane)`
    lane-0-only residual.
24. **No SIMD4i** (`project_simd4i_decision`) — real int primitive is
    `VectorDot`→SDOT; wasm int `VectorBinOp` latent miscompile.
25. **PtrAdd byte scaling** (`project_tier_promotion_default_llvm`) — intermittent
    load crash; CallIndirect widths miscompile.
26. **captured method call in loop SIGSEGV** (`bugs_captured_method_call_in_loop_sigsegv`)
    — closure over loop-var (fixed); residual value-type `n=n+1`.

### F. Ownership / trap-stub / misc
27. **trap-stub cascade** (`bugs_trap_stub_cascade`) — forward-ref/unresolved →
    trap stub; `RAYZOR_DUMP_FN_PTRS=1` + crash-PC→fn-ptr map to diagnose.
28. **cache-hit fast path bypasses fixups** (`bugs_cache_hit_fast_path_bypasses_fixups`)
    — use `--no-cache` when editing imported modules.
29. **rope interleaved for GGUF** (`bugs_rope_interleaved_for_gguf`) — THE GGUF
    coherence bug; interleaved vs half-split rotation. Relevant to the rope port.
30. **f16 carry decay** (`13dff852`) — f16 scale decode carry in flash.
31. **pool data-flow gap** (`bugs_pool_data_flow_gap`) — transient activations
    never reach `rayzor_tensor_free` (InsertFreePass gap) — relevant to the
    lifecycle redesign.

### Discipline (rules, not bugs)
- Never `Array<Float>`/`Float`-heavy loops in kernels → raw `Bytes` + `Mem`.
- `--no-cache` when editing imported modules; `--llvm --release` for perf.
- Verify codegen by **disasm**, never IR/source (SDOT/hsum).
- Bit-exact vs the Rust oracle before deleting any Rust op.
- No silent dispatch fallthrough — unresolved FQN = hard error.
- `pkill -9 -f '[r]ayzor'` + assert 0 lingering around benches; capture
  `Code Helper (Plugin)` CPU (co-runner confounds M1 wall-clock).

## 6. Sequencing vs the in-flight work
Insight: most env-var decoupling is **entangled with the move** — the inference
env reads in Rust core live in the kernels that relocate (`NEON_SILU`,
`USE_SDOT`, `LEGACY_KERNEL`, `PREFILL_MORSELS`, `KERNEL_TIMING`), and even
`RAYZOR_HAXE_MATMUL` (`worker_pool.rs`) sizes the pool the FFI kernels use. So
the move REMOVES most of the inference env from core structurally. Doing a
piecemeal env rename first is largely redundant.

1. **Move `Tensor`/`QTensor` → `nue`** (Phase C relocation; kernels stay
   FFI-in-plugin initially — call-neutral). This is the primary structural work
   and it drains the inference env + downstream doc-context out of core.
2. **Decouple the residual core env** that is NOT part of the move: the general
   runtime (`worker_pool` `RAYZOR_HAXE_MATMUL` sizing — marked `TODO(move)`;
   caller declares width via a hint, core reads no inference flag), the 4
   deprecated pool reverts (`NO_CALLER_BAND`/`STATIC_BANDS`/`NO_PARK`/
   `LEGACY_POOL`), compile-gate the ~8 diagnostics.
3. **`NUE_` rename** the surviving inference config (Nue-side + plugin) with
   `RAYZOR_*` aliases. Pool vars already done (`1fab8e78`).
4. **Platform backends** (Phase D) — Apple Accelerate/AMX + x86 VNNI behind the
   backend interface.
5. **Pure-Haxe op-math + data structure** (Phases A/B) — the long tail; delete
   Rust as each op validates. The "pick up later" bulk.
6. **Rayzor runtime doc-purity pass** — AFTER the move: make every core doc
   downstream-context-free (no `Nue`/inference/decode/tok-s narrative, no
   ephemeral commit IDs). `worker_pool.rs` module doc done as a first pass; the
   rest (`tensor.rs`, `quant.rs` leave with the move; `tensor_pool`, `profile`,
   etc. get cleaned) follows.
