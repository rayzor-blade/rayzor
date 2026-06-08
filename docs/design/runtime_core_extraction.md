# runtime-core Extraction — Phase 1 Migration Detail

Status: in progress (2026-06-08)
Parent: [wasm_runtime_parity.md](wasm_runtime_parity.md) Phase 1

## Premise

The native `runtime/` crate and the WASM `runtime-wasm/` crate currently
have **zero shared algorithm code**. The 2.5× decode-tok/s and 30× peak-RSS
improvements landed this session (commits `c5ab136`, `4001079`, `7835a26`,
`a241cc9`, `acb80e5`, `7b72ba8`) all live exclusively in `runtime/` and
will never reach the WASM target unless someone ports them by hand.

This extraction creates a third crate, `runtime-core/`, that is `no_std +
alloc`-compatible and holds the architecture-portable compute kernels.
Both `runtime/` and `runtime-wasm/` depend on it. Per-arch SIMD lives
behind `cfg(target_arch)` gates with scalar fallbacks. One source of
truth, three deployment targets.

## Inventory summary

Workflow `wf_f66b9e1b-19f` (2026-06-08) scanned 11 788 LOC across
`runtime/src/{tensor_simd,quant,tensor}.rs` and classified every function
by `no_std + alloc` candidacy.

| File | LOC | Pure-compute fns | Native-only fns | Shared types |
|---|---:|---:|---:|---:|
| `tensor_simd.rs` | 1 048 | **18** | 0 | 0 |
| `quant.rs` | 4 825 | ~30 | ~26 (FFI + pool + threading) | Q4KBlock, Q8Block, Q8KBlock, Q4KMBlock |
| `tensor.rs` | 4 899 | ~19 | ~70 (FFI + alloc + pool + accessors) | RayzorTensor, DTYPE_*, DEVICE_*, FP8_*_LUT |

The shared types are all `no_std`-compatible: `core::sync::atomic::AtomicUsize`
works under `no_std`. The boundary between `runtime-core/` and `runtime/`
sits at **allocation**, not refcount or struct layout.

## Crate skeleton

```
runtime-core/
├── Cargo.toml         no_std + alloc; deps: half (default-features = false)
└── src/
    ├── lib.rs         #![cfg_attr(not(test), no_std)]; pub mod simd, quant, tensor;
    ├── simd/
    │   ├── mod.rs
    │   └── tensor_f32.rs    18 functions migrated from runtime/src/tensor_simd.rs
    ├── quant/                (Step 2-3)
    │   ├── mod.rs
    │   ├── types.rs          Q4KBlock, Q8Block, Q8KBlock, Q4KMBlock
    │   ├── q4_k_m.rs         decode/dequant/encode block ops
    │   ├── q6_k.rs           dequant_q6_k_block
    │   ├── q8_k.rs           quantize_x_block_q8, quantize_row_q8_K, prepare_x_q8k_blocks(_into)
    │   ├── sdot.rs           dot_q4_k_q8*, dot_q6_k_q8, sdot_enabled_runtime
    │   └── matmul.rs         qmatmul_chunk_impl(_sdot_q4km), int8_matmul_f32, q4_k_m_matmul_f32
    └── tensor/                (Step 4-5)
        ├── mod.rs
        ├── types.rs          RayzorTensor (excl. alloc methods), DTYPE_*, DEVICE_*
        ├── dtype.rs          dtype_size, load_f32_at, store_f32_at, fill_dtype, pool_alloc_bytes
        ├── fp8.rs            E4M3/E5M2 encode + decode + const LUTs
        ├── binop.rs          prepare_binop, tensor_binop_*, tensor_unary
        ├── flash_attn.rs     flash_attn_decode_one_qhead
        ├── rope.rs           rope_table
        └── topk.rs           recent_contains
```

## Sequencing

Migration runs in six discrete steps, each one a separate commit with full
`cargo test` + Paris MATCH + decode-perf neutrality check before moving on.

### Step 1 — Create `runtime-core/` crate skeleton  *(landing)*

- Add `runtime-core/Cargo.toml`, `runtime-core/src/lib.rs`.
- Add `runtime-core` as workspace member.
- Empty modules so the workspace builds clean.
- Verify: `cargo build --release` succeeds, no warnings.

### Step 2 — Migrate `tensor_simd.rs`  *(landing)*

- Whole file moves to `runtime-core/src/simd/tensor_f32.rs`.
- `std::arch::*` → `core::arch::*`, `std::slice::*` → `core::slice::*`.
- `runtime/src/tensor_simd.rs` becomes a 3-line `pub use` shim; every
  existing `use crate::tensor_simd::*` keeps working.
- Verify: workspace `cargo build`, `cargo test -p rayzor-runtime-core`,
  Paris MATCH, decode tok/s (>70 baseline) within noise.

### Step 3 — Migrate `quant.rs` types + block ops

- `Q4KBlock`, `Q8Block`, `Q8KBlock`, `Q4KMBlock` → `runtime-core/src/quant/types.rs`.
- `decode_q4_k_block`, `dequant_q4_k_block`, `q4_k_get_scale_min`,
  `pack_q4_k_scales`, `quantize_block_q4_k_m` → `runtime-core/src/quant/q4_k_m.rs`.
- `dequant_q6_k_block` → `runtime-core/src/quant/q6_k.rs`.
- `quantize_x_block_q8`, `quantize_row_q8_K`, `prepare_x_q8k_blocks(_into)`
  → `runtime-core/src/quant/q8_k.rs`.
- `sdot_enabled_runtime` → `runtime-core/src/quant/sdot.rs` (the
  env-var-reading `sdot_enabled` stays native).

### Step 4 — Migrate the SDOT kernels (the perf-critical c5ab136 set)

- `dot_q4_k_q8`, `dot_q4_k_q8_kblock`, `dot_q4_k_q8_kblock_llamacpp`,
  `dot_q4_k_q8_kblock_2`, `dot_q6_k_q8`, `vec_dot_q4_K_q8_K`
  → `runtime-core/src/quant/sdot.rs`.
- These are the kernels that delivered the +15.4% (c5ab136) and +3.04%
  (acb80e5) wins. Verify decode tok/s **strictly** post-migration.

### Step 5 — Migrate matmul chunk implementations

- `qmatmul_chunk_impl_sdot_q4km`, `qmatmul_chunk_impl`, `int8_matmul_f32`,
  `q4_k_m_matmul_f32`, `dot_f32_simd`, `dot_f32_avx2_fma`, `qmatmul_prep`,
  `x_is_contiguous`, `x_tensor_data_ptr` → `runtime-core/src/quant/matmul.rs`.
- The threaded entry points (`rayzor_tensor_matmul_qt_t_f32_threaded` et al.)
  stay native — they own the `worker_pool::global().parallel_rows` dispatch.
- Native wrappers become thin shells calling `runtime_core::quant::matmul::*`.

### Step 6 — Migrate `tensor.rs` pure-compute helpers

- `dtype_size`, `load_f32_at`, `store_f32_at`, `fill_dtype`, `pool_alloc_bytes`
  → `runtime-core/src/tensor/dtype.rs`.
- FP8 encode/decode + `FP8_E4M3_LUT`/`FP8_E5M2_LUT` → `runtime-core/src/tensor/fp8.rs`.
- `prepare_binop`, `tensor_binop_row_broadcast`, `tensor_binop_scalar`,
  `tensor_unary` → `runtime-core/src/tensor/binop.rs`.
- `flash_attn_decode_one_qhead` → `runtime-core/src/tensor/flash_attn.rs`
  (the parallelisation wrapper stays in native `tensor.rs`).
- `rope_table` → `runtime-core/src/tensor/rope.rs`.
- `recent_contains` → `runtime-core/src/tensor/topk.rs`.
- `RayzorTensor` struct layout + DTYPE/DEVICE constants → `runtime-core/src/tensor/types.rs`
  (allocation methods stay native via `impl` block in `runtime/src/tensor.rs`).

## Verification gates (every step)

- `cargo build --release` workspace clean (including `--features llvm-backend`).
- `cargo test -p rayzor-runtime-core` green.
- `cargo test -p rayzor-runtime` green (or whatever native tests existed before).
- `cargo fmt --check`, `cargo clippy --workspace -- -D warnings` clean.
- 128/128 Haxe regression preserved.
- Llama 3.2 1B Q4_K_M decode tok/s on canonical Voronoi 808-token prompt:
  >70 tok/s steady, RAYZOR_KERNEL_TIMING unset. No regression past noise.

## Non-goals for Phase 1

- **No wasm32-simd128 kernel additions.** Adding the wasm SIMD path is
  Phase 2 work — the scalar fallback already compiles correctly under
  `cfg(target_arch = "wasm32")`.
- **No FFI surface changes in runtime/.** Every `extern "C"`
  `rayzor_tensor_*` symbol behaves identically pre and post extraction.
- **No `runtime-wasm/` wiring.** The WASM crate doesn't yet depend on
  `runtime-core`; that's Phase 2 (Tensor lifetime + F32 matmul on WASM).
- **No removal of the `runtime/` re-export shims.** Existing callers
  inside `runtime/` keep their `crate::tensor_simd::foo` paths. We can
  inline the shims later when call sites get touched for other reasons.

## Why Step 2 is risk-low

The full set of decode-perf wins this session lives in `quant.rs`
(Steps 3-5). `tensor_simd.rs` (Step 2) is the safest first chunk:

- Zero native-only functions in the file.
- No threading, no globals, no allocator coupling.
- Single external dep (`half`) is already `no_std`-compatible.
- All 18 functions are slice-in slice-out with debug-assert length contracts.
- Comprehensive unit tests already exist (10 `#[test]` functions covering
  every kernel) and move with the code.

If Step 2 introduces a perf regression, it's purely a build/link concern
(LTO across crate boundaries) — fixable with `lto = "fat"` and
`codegen-units = 1` in the workspace release profile.
