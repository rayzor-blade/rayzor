# Nue engines — capability matrix & selection

Nue's dispatch principle: **the engine is a function of (platform × model ×
phase), resolved at load by detection — never a global choice.** The platform
comes from the system, the model's needs from its own metadata, accelerator
availability from probing. Zero configuration is the intended mode; the
resolved engine is echoed at load (`[embed] engine=...`).

## Encode (bert-family)

| Engine  | Backend                          | Platform | Needs                                   | M1 measured        | Cosine vs ST golden |
|---------|----------------------------------|----------|------------------------------------------|--------------------|---------------------|
| `ane`   | CoreML runtime, CPU+NeuralEngine | macOS    | `<stem>.encoder_s{S}.mlmodelc` artifacts | **~830 sent/s**    | 0.9999443           |
| `graph` | BNNSGraph (fused, CPU-only)      | macOS    | same artifacts                           | ~400 sent/s        | 0.9999628           |
| `amx`   | Accelerate f16 per-op GEMM       | macOS    | —                                        | ~253 sent/s        | 0.9999981 (bit-class)|
| `int8`  | VNNI `vpdpbusd` GEMM (SDOT on arm)| any     | `hidden % 16 == 0`                       | 127 sent/s (NUC)   | 0.9920855           |
| `haxe`  | portable F32 SIMD                | any      | —                                        | 59 sent/s (NUC)    | 0.9999983 (bit-class)|

Auto policy: macOS → `ane` → `graph` → `amx`; elsewhere → `int8` → `haxe`.
Accuracy note: the fp16/int8 engines shift cosine within the ≥0.99 gate by
design — force `amx`/`haxe` when a bit-class baseline is needed.

## Prefill vs decode (llama-family)

- **Prefill** is the encode-shaped phase (compute-bound batch GEMM) and will
  join this selector when the fused prefill graph lands. On unified memory the
  ANE-prefill → CPU-decode split is zero-copy: the graph writes the KV cache
  into the same DRAM the decode loop reads.
- **Decode** is deliberately NOT selectable onto accelerators: batch-1 GEMV is
  weight-bandwidth-bound and per-token dispatch latency erases any gain.
  Decode always runs the CPU kernel path; only the SIMD flavor differs by
  platform (NEON SDOT / x86 `vpdpbusd`).

## Selection

1. **Detection (default, zero config).** Probes in best-first order with
   portable fallback. When graph/ane artifacts are absent on a Mac, a hint is
   printed — they are a build product (like the gguf itself), authored once
   per model:

   ```
   mlvenv/bin/python rayzor-tensors/examples/bert_graph_author.py \
       <model.gguf> <outdir> 128 256 512
   xcrun coremlc compile <outdir>/<stem>.encoder_s{S}.mlpackage <outdir>
   # place the .mlmodelc bundles next to the gguf
   ```

   Dims (hidden/layers/heads/ffn) are read from the gguf's own `bert.*`
   metadata; artifacts are named by the gguf stem so models share a directory.

2. **Escape hatch:** `NUE_ENGINE=haxe|amx|int8|graph|ane` (legacy
   `RZT_EMBED_ENGINE`) — for A/B measurement and pinning. An unsatisfiable
   request falls back down the same ladder and says so.

## Landmines (hard-won)

- A Rust binary calling BNNSGraph must link Accelerate **explicitly** —
  flat-namespace `dynamic_lookup` lazy-loads its C++ statics mid-call and
  aborts in libc++abi (macOS 26).
- New FFI symbols must be added to `rayzor-tensors`' `plugin_init` symbol
  table, and the **platform dylib rebuilt** (`cargo build -p rayzor-tensors`)
  — a stale `.so`/`.dylib` fails at JIT link with `can't resolve symbol`.
- Never string-concat an x-module extern's return value (mistypes as an
  object; concat dereferences the raw value). Compares/assigns are safe.
- mlprogram argument order is alphabetized — resolve positions by name.
- Keep engine-selection logic inside the module that uses it: adding a new
  module shifts function IDs and can land startup on a pre-existing trap stub
  (reflect-renumbering family).
