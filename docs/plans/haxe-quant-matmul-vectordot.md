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

## Phase 1 — DONE (2026-06-20): SIMD4i32 works native + wasm.

`SIMD4i32` (i32x4) is wired end-to-end and PASSES on both targets (make/splat/
+/-/*/get(const+runtime)/sum = 50/100/11/14; test_simd4i32.hx). It took TWO
fixes that were jointly required:

1. **`elem_ty` on the vector IR instructions.** `VectorExtract/Insert/Reduce`
   carried no element type — the wasm backend inferred it from `register_types`,
   which is LOST when a vector wrapper is inlined → an inlined integer reduce
   emitted `f32x4.extract_lane` on an i32 vector (i32 lanes read as floats → 0).
   Added `elem_ty: IrType` to all three (the MirBuilder builders already received
   it); the wasm backend reads it directly. ~12 files; non-wasm match arms just
   `..`-ignore it. Survives inlining (remap_instruction clones it).

2. **SIMD instance-method dispatch.** hir_to_mir hard-coded a `rayzor_SIMD4f`
   class hint for ANY vector receiver, so `SIMD4i32.sum()/get()` dispatched to
   the SIMD4f (f32) wrappers. Native masked it (Cranelift reduces/extracts by the
   SSA value type = i32x4 → correct value by accident); wasm exposed it (uses
   `elem_ty` from the f32 wrapper → 0). Added `simd_vector_class(ty)`:
   i32x4 → `rayzor_SIMD4i32`, else `rayzor_SIMD4f`, used at both hint-assignment
   sites + the SIMD direct-lookup dispatch.

No regressions: test_simd_e2e 17/17, test_tensor_e2e 20/20, SimdDemo + getlane2
f32 unchanged, native suite 139 pass. Debugging tool: `RAYZOR_DUMP_WASM` +
`wasm-tools print` (the f32x4-on-i32 disasm cracked it). Phase 0 element-aware
wasm vector arms committed (87a14a6e).

## Phase 1 — Integer SIMD types + stdlib surface

- Add abstracts: `SIMD16i8` (i8x16 dot operands), `SIMD16u8` (u8x16 dequant
  masks/shifts), `SIMD4i32` (the i32x4 accumulator — the only place 4×i32 belongs).
- Wire each → `IrType::vector(<int>, N)` in hir_to_mir (mirror the SIMD4f arm).
- Add stdlib MIR wrappers mirroring SIMD4f_* (`systems.rs:787-1189`):
  `load/store/splat/make/extract/insert` + integer `and/or/shl/shr` + widening
  `extend_low/high`.
- **Gate:** load 16×i8, and-mask `0x0F`, `shr 4`, store — bit-correct native+wasm.

## Phase 2 — DONE (2026-06-20): VectorDot lands SDOT on all backends.

`SIMD16i8` (i8x16) + `SIMD4i32.dot(acc, a, b)` work native + wasm + LLVM with
identical values (test_simd_dot.hx: 96/192/560). The new `VectorDot { dest, acc,
a, b }` IR op lowers to the SDOT path on **every** backend:
- **Cranelift (native):** the exact `swiden_low/high → imul → iadd_pairwise →
  swiden_low/high → iadd_pairwise → iadd` CLIF tree — CLIF-dump-verified — which
  is precisely what PR #13640's FEAT_DotProd ISLE rule folds to `sdot` (same
  cranelift that yields 24 sdot in production wasm).
- **wasm:** `i32x4.relaxed_dot_i8x16_i7x16_add_s` (dump-verified, 26×) → wasmtime
  47 → SDOT.
- **LLVM:** `sext<16xi8→i32> → mul → 4-way strided shuffle-reduce → add acc` —
  the canonical idiom LLVM's AArch64 backend folds to SDOT. (LLVM is NOT optional;
  it's where the Rust kernel itself gets SDOT — implemented properly, not stubbed.)
- **c/interpreter:** correct scalar fallbacks (interpreter bails to JIT).

Wiring: VectorDot in instructions.rs (+ dest/replace_dest/operands/replace_uses)
+ MirBuilder::vector_dot + Opcode::VectorDot. SIMD16i8.hx (splat/load), SIMD4i32.dot,
hir_to_mir type-map (vector(I8,16)), runtime_mapping VecI8x16 + dot16 + SIMD16i8
methods, systems.rs wrappers (dot16/splat/load). No regressions (simd_e2e 17/17,
simd4i32 + f32 unchanged, native suite — only known-flaky test_copy/test_exception).

## (original) Phase 2 — The `VectorDot` fused-dot IR op *(leverage point)*

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
