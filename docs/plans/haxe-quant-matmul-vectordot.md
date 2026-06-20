# Plan: pure-Haxe quantized matmul via a fused integer-dot primitive

**Status:** proposed (2026-06-20). Phase 0 in progress.

**Goal:** move the Q4_K_M / Q6_K × Q8_K matmul dot into pure Haxe, then fuse
dequant + dot + scale across the (currently FFI) boundary — for superior **wasm**
decode latency. Native is already at NEON-SDOT parity; the win is removing the
FFI fusion barrier (the documented #1 wasm decode lever — "CALL never inlined on
wasm").

**Why this is feasible now (not a parity-replication exercise):** the SDOT
lowering already ships in wasmtime 47 / cranelift 0.134 (darmie's PR #13640) —
wasm `i32x4.relaxed_dot_i8x16_i7x16_add_s` → AArch64 SDOT, and native
`swiden+imul+iadd_pairwise` CLIF → SDOT via a priority-8 ISLE rule. Production
`rayzor_runtime_wasm.wasm` objdumps to 24 sdot / 0 smull. So this plan only has
to make rayzor's *own* backends **emit** the dot op; the downstream lowering is
done. The "won't reach SDOT" risk does not exist.

**Why not "SIMD4i":** a 4×i32 elementwise vector is the wrong shape. The quant
hot path is a **16×i8 input → 4×i32 accumulator FUSED widening dot**, not an
i32x4 elementwise op. A 4×i32 type only models the *accumulator*. The right
primitive is an `i8x16` type + a `VectorDot` op. (See
`memory/project_simd4i_decision.md`.)

---

## Grounding (investigation file:line anchors)

- IR `IrType::Vector{element,count}` is already element-generic — `compiler/src/ir/types.rs:49`.
- Vector IR instructions carry `vec_ty`, not f32-specialized — `compiler/src/ir/instructions.rs:375`.
- Native Cranelift already lowers integer elementwise + reduce (iadd/isub/imul/sdiv/band/bor/bxor over I8X16..I64X2) — `compiler/src/codegen/cranelift_backend.rs:4442,5135`. No fused dot.
- WASM backend HARDCODED to f32x4 for all vector arithmetic — `compiler/src/codegen/wasm_backend.rs:~3611-3760`. Integer `VectorBinOp` silently → `F32x4Add` (latent bug). `VectorLoad/Store` + V128 typing already element-agnostic — `:3771,:4501`.
- wasm_linker maps full int + relaxed-simd opcode set incl `I32x4DotI16x8S` (`:2034`) and relaxed dots (`:2198-99`) — reachable once codegen emits them.
- Front-end: only `rayzor::SIMD4f → vector(F32,4)` wired — `compiler/src/ir/hir_to_mir.rs:~21810`. VectorBinOp short-circuit element-generic — `:19259`. SIMD4f wrappers f32-only — `compiler/src/stdlib/systems.rs:787-1189`. `SIMD4f_dot` = Mul+reduce (not a fused dot) — `:~1074`.
- Quant kernel shape: 16×i8 → i32x4 fused dot. Q4: nibble unpack (`and 0x0F`, `shr 4`) → two i8x16 → relaxed_dot/vdotq → i32x4 acc → `i32x4_mul` scale → hsum once/super-block → f32 fold `d·Σ(scale·dot) − dmin·Σ(min·bsum)`. Q6: + 6-bit reconstruct (`ql | qh<<4`). Activation Q8_K: `[i8;256]` qs + `[i16;16]` bsums + f32 d.
- Kernel sources: `runtime-core/src/quant/sdot.rs` (NEON), `q4_k_m.rs` / `q6_k.rs` (wasm SIMD128), `types.rs` (block layouts).
- SDOT shipped: PR #13640, wasmtime 47/cranelift 0.134 (rev 8cb28bc), production 24 sdot. lever-1 (defer hsum to vector domain) landed 66beadf (~9%, ~30 tok/s wasm; kernel-source matmul opt exhausted).

---

## Phase 0 — Fix the wasm integer-vector miscompile *(correctness; standalone; do regardless)*

`wasm_backend.rs:~3611-3760` hardcodes f32x4 for all vector arithmetic, so any
integer `VectorBinOp` silently emits `F32x4Add` and `and/or/xor` traps.
Everything below sits on this.

- Make `VectorSplat / VectorBinOp / VectorExtract / VectorInsert / VectorUnaryOp /
  VectorMinMax / VectorReduce` branch on `vec_ty` element type → emit
  `I8x16 / I16x8 / I32x4 / I64x2` ops.
- Pick `ExtractLaneS` vs `ExtractLaneU` from element signedness.
- `VectorLoad/Store` + V128 local-typing unchanged (already element-agnostic).
- **Gate:** Haxe test doing integer vector `add/mul/and/shr` returns correct
  values on wasm (today wrong) and matches native.

## Phase 1 status (2026-06-20): native WORKS; wasm BLOCKED on an IR-type gap

A first integer type `SIMD4i32` (i32x4) was wired end-to-end (abstract +
hir_to_mir type map + tast_to_hir `@:op` skip + runtime_mapping + systems.rs
wrappers + `VecI32x4` descriptor). Result:
- **Native: PASS** — `make/splat/+/-/*/get(const+runtime)/sum` all correct
  (50/100/11/14). Proves the design + wiring + Phase 0 are right.
- **WASM: returns 0.** ROOT CAUSE (objdump-confirmed via `RAYZOR_DUMP_WASM`):
  `VectorExtract`/`VectorReduce`/`VectorInsert` carry **no element type** in the
  IR — the wasm backend infers it from `register_types[vector]`. That works for
  a *called* wrapper (the param type is in the signature) but **breaks when the
  wrapper is inlined into the caller**: the inlined vector operand's reg has no
  `vector(I32,4)` entry in the caller's `register_types`, so the backend
  defaults to F32 and emits `f32x4.extract_lane` on an i32 vector → reads i32
  lanes as floats → ~0 → truncates to 0. Main had 16 `f32x4.extract_lane`, zero
  `i32x4.extract_lane`. (Native is immune — Cranelift tracks SSA value types.)
  `InlineHint::Never` on the wrappers did NOT prevent it (sum/extract still
  inlined by a lower pass), confirming the fix must be in the IR, not inlining.

**THE FIX (prerequisite for any integer SIMD on wasm, incl. the quant kernel):**
add an explicit `elem_ty: IrType` field to `VectorExtract`, `VectorInsert`,
`VectorReduce` (the MirBuilder builders ALREADY receive it — `vector_extract(vec,
idx, elem_ty)`, `vector_reduce(op, vec, elem_ty)` — they just don't store it).
Then the wasm backend reads `elem_ty` directly instead of `vec_elem_of_reg`.
Blast radius ~12 files (instructions.rs + mir_builder.rs + match arms in
cranelift/wasm/llvm/c/interpreter/tiered/dump/optimization/vectorization/
builder) — each non-wasm arm just binds-and-ignores the field. Do this as its
own careful pass with all-backend verification BEFORE landing SIMD4i32.

`vec_elem_of_reg` (wasm_backend) + the param-signature fallback are still correct
and useful (they fix the *called*-wrapper case); the IR field makes the inlined
case robust too. Phase 0 (element-aware wasm vector arms) is committed and solid.

## Phase 1 — Integer SIMD types + stdlib surface

- Add abstracts: `SIMD16i8` (i8x16 dot operands), `SIMD16u8` (u8x16 dequant
  masks/shifts), `SIMD4i32` (the i32x4 accumulator — the only place 4×i32 belongs).
- Wire each → `IrType::vector(<int>, N)` in hir_to_mir (mirror the SIMD4f arm).
- Add stdlib MIR wrappers mirroring SIMD4f_* (`systems.rs:787-1189`):
  `load/store/splat/make/extract/insert` + integer `and/or/shl/shr` + widening
  `extend_low/high`.
- **Gate:** load 16×i8, and-mask `0x0F`, `shr 4`, store — bit-correct native+wasm.

## Phase 2 — The `VectorDot` fused-dot IR op *(leverage point)*

- New IR instruction `VectorDot { acc, a, b, widen }` (`i8x16→i32x4` and
  `i16x8→i32x4`); add `MirBuilder::vector_dot`.
- **Cranelift arm:** emit the `swiden_low/high + imul + iadd_pairwise + iadd` CLIF
  tree that PR #13640's priority-8 ISLE rule contracts to SDOT. Objdump-verify
  `sdot`, not `smull`, on a FEAT_DotProd host.
- **Wasm arm:** emit `i32x4.relaxed_dot_i8x16_i7x16_add_s` (→ SDOT under wasmtime
  47), with `i16x8.extend + i32x4.dot_i16x8` fallback for no-relaxed-simd. Linker
  already maps both.
- Expose Haxe intrinsic, e.g. `SIMD4i32 dot16(acc, a:SIMD16i8, b:SIMD16i8)`.
- **Gate:** standalone Haxe 16×i8 dot == scalar reference, bit-exact native+wasm;
  objdump confirms SDOT in both native machine code and JIT'd wasm.

## Phase 3 — Port the Q4_K_M dot kernel to Haxe

Template off `q4_k_m.rs:vec_dot_q4_K_q8_K_simd128` (cleaner than NEON intrinsics).

- Haxe block structs matching `types.rs`: `Q4KMBlock`, `Q8KBlock`.
- Loop: nibble-unpack → two `SIMD16i8` → `dot16` accumulate over 8 sub-blocks →
  i32x4 scale in vector domain → **defer hsum to once/super-block** (lever-1,
  proven +9% at 66beadf) → f32 fold.
- **Gate:** bit-identical dot vs Rust kernel on real GGUF blocks; France→Paris
  greedy match; decode A/B vs FFI kernel.

## Phase 4 — Fuse across the removed FFI barrier *(the superior-latency win)*

- Inline the Haxe dot into the qmatmul band loop so dequant + dot + scale +
  surrounding elementwise fuse (no FFI call, no marshaling — the L1 lever).
- Measure in-guest wasm decode tok/s vs FFI baseline (target: beat ~30 tok/s).
- Q6_K follows the same template + 6-bit reconstruction.

## Phase 5 — *(parallel/optional)* close the Cranelift-vs-LLVM floor upstream

Residual non-dot gap (no `addv` hsum tax, weaker scheduler/regalloc, 2-acc ILP
ceiling — the 30→90 tok/s gap). Upstream cranelift (addv-fusion for i32x4
reductions + aarch64 regalloc/scheduling), same muscle as #13640, against the
`8cb28bc` checkout.

---

## Cross-cutting risks / gates

- **SDOT-not-emitted:** objdump every target; the CLIF pattern must match the
  ISLE rule exactly (#13640 makes it reachable, not automatic).
- **relaxed_dot determinism:** exact only in-range. Q4 nibbles 0..15 and Q6 6-bit
  0..63 both fit the i7x16 operand (≤127); activations i8 → bit-identical across
  runtimes (memory-confirmed for Q4; **verify Q6's 0..63 operand** as a gate).
- **Don't replay native-only ILP tricks:** 2-row tiling *regressed* on wasm
  (cranelift regalloc spills). Expect Phase 4 = parity-to-modest until Phase 5.
- **Validation discipline:** bit-exact dot before any perf claim; decode A/B with
  `pkill -9` hygiene; never attribute flaky SIGTRAP (test_copy_clone_identity /
  test_exception) to a change without an N≥40 baseline.

## Sequencing

Phase 0 is independent correctness (land first). Phases 1–2 build the primitive.
Phases 3–4 are the kernel move + latency payoff. Phase 5 is the upstream floor
that gates "superior vs parity" on wasm.

## Build/test notes

- `rayzor` binary is the ROOT package: build with
  `cargo build --release --features llvm-backend -p rayzor --bin rayzor`
  (`-p compiler` compiles the lib but does NOT relink the binary).
- Run a Haxe file: `./target/release/rayzor run <file>.hx [--wasm] --no-cache`.
- Suite: `./run_haxe_tests.sh` (native only). SIMD testbed:
  `cargo run --release --features llvm-backend -p compiler --example test_simd_e2e`.
- Kill orphans before benches: `pkill -9 -f "rayzor run"; pkill -9 wasmtime`.
