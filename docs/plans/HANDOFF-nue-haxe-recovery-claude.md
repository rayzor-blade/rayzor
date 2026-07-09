# HANDOFF: Nue pure-Haxe decode recovery for Claude

Audience: Claude or any next agent picking up Nue/Rayzor performance work.
This is the state after the NUC investigation and the Mac regression recovery
pass. Read this before changing llama-chat defaults, flash attention, tiering,
or the spin pool.

## Non-negotiables

- The goal is the pure-Haxe path: `RAYZOR_HAXE_MATMUL=1`,
  `RAYZOR_HAXE_FLASH=1`, Q8 KV cache, and Haxe-side decode kernels.
- Do not "fix" performance by turning Haxe flash off or falling back to Rust
  FFI matmuls. Rust FFI remains a reference path, not the answer.
- Do not make AOT the default tier for `rayzor run`. MCJIT/tier-up must work;
  AOT is an opt-in shipping/cold-start mode.
- Do not trust tiny-token runs for throughput. Use the long Voronoi prompt or
  a similarly long decode, and separate profiled from unprofiled runs.
- Do not force Mac worker/flash/batch constants from scripts. Let the runtime
  choose unless an env var is explicitly set for an A/B.
- Generated `.rzb` bundles are artifacts. Do not commit them.

## What happened

The NUC work improved x86 pure-Haxe viability but introduced several risky
experiments into the default Mac path. The visible symptoms were:

- Mac decode dropped from the historical 75-85 tok/s range into the 50-60
  tok/s band, with very high variance.
- TTFT often climbed to ~1s on prompts that previously reached sub-second.
- Long generations sometimes produced corrupted text.
- Shape corruption surfaced as impossible tensor shapes in residual add:
  `rayzor_tensor_add_into: shape mismatch at dim 0`.
- `--stats` and profiling runs were compared against unprofiled runs, which
  hid the real signal.

The main root causes we isolated:

- The shifted-query Q8 flash experiment became too close to the default path.
  It uses `q + 128`, `dotI8U8`, and subtracts `128 * sum(k)`. It is useful for
  x86 VNNI experiments, but a miss in this path corrupts every decode step.
- Forced fused matmul was not stable as a default. It reduces dispatch count
  but regressed full llama-chat runs compared to split kernels.
- `start_interpreted=true` plus eager LLVM/stat instrumentation polluted the
  baseline. With `--llvm`, Beadie route counters can stay at zero because
  direct native entry bypasses the profile dispatcher.
- Visible token string streaming still costs wall time. Silent streaming can
  reach much higher throughput, but visible streaming must not block decode in
  the long term.

## Recovery changes currently in tree

### FlashDecode

`nue/nue/transformer/FlashDecode.hx`

- Haxe flash remains enabled when `RAYZOR_HAXE_FLASH=1`.
- Signed Q8xQ8 attention is the production default again.
- The shifted-query VNNI path is opt-in through
  `RAYZOR_HAXE_FLASH_SHIFTED_Q=1`.
- Batch flash/speculative verification is disabled on the default path:
  `batchMax()` returns `1`, and `decodeBatch()` returns `null`.
- Flash bands still use the spin pool by default. `RAYZOR_HAXE_FLASH_POOL=0`
  is only an A/B escape hatch.

### Q4Matmul and fused gates

`nue/nue/Q4Matmul.hx`

- `RAYZOR_HAXE_FUSED_MATMUL` is opt-in. Unset means split kernels.
- Quantize profiling only records timing when the pool is in profiling mode.
- Quantize parallelism has a minimum work threshold tied to worker count.
- The fused row-space implementation remains in tree for controlled testing,
  but should not be treated as production until it is bit-stable and faster in
  full llama-chat, not just microbenchmarks.

### Llama hot path

- `nue/nue/transformer/LlamaBlock.hx` is a concrete Llama block that avoids
  interface dispatch through the hot path.
- `nue/nue/sampling/LocalTempSampler.hx` moved the concrete top-k +
  repetition-penalty sampler out of `Main.hx`, avoiding trap-stub cascades from
  importing generic sampler classes.
- `GenerationLoop` is specialized to `LlamaModel` and `BPETokenizer` and uses
  `forwardLastLogits()` for decode steps.
- `Main.hx` has TTFT measurement, optional silent stream, and buffered visible
  streaming. This is a mitigation, not the final async streaming design.

### SpinPool and topology

`compiler/haxe-std/rayzor/concurrent/SpinPool.hx`
`runtime/src/topology/*`

- Pool profiling is env-gated and no longer charges every dispatch by default.
- `CpuTopology.bindPerformance()` exists for runtime/platform affinity.
- macOS performance affinity is opt-in through `RAYZOR_MAC_PERF_AFFINITY`;
  do not force it from scripts.
- macOS relax default is off because the Haxe-level relax wrapper costs too
  much in hot joins. Linux/x86 default remains relax-on for thermal and
  memory-order reasons.
- Apple Silicon gets a higher runtime-derived spin budget so workers do not
  park between short decode dispatches.

### CLI and bundle execution

`src/main.rs`
`compiler/src/codegen/tiered_backend.rs`

- `.rzb` execution now honors `--llvm`, matching source execution.
- `--tier-start-interpreted` and `--tier-promotion` are CLI overrides, so
  shipped bundles do not need accompanying `rayzor.toml` files.
- `RAYZOR_DUMP_TIERS=1 --stats` can dump per-function tier listings.
- AOT tiering is opt-in via `RAYZOR_LLVM_TIER_AOT=1`; it is not the default.

### Compiler/runtime safety

`compiler/src/codegen/llvm_jit_backend.rs`

- Missing LLVM SSA values now error instead of silently lowering to null. The
  old behavior could turn a compile-time dominance bug into runtime tensor heap
  corruption much later in residual adds.

`runtime/src/haxe_string.rs`

- Haxe string handling was touched during streaming work. If visible streaming
  still blocks decode, fix the stdout/string path directly rather than
  disabling flash or matmul.

## run_bundle defaults after recovery

`nue/examples/llama-chat/run_bundle.sh`

The script now enables the intended Haxe inference path but avoids overriding
experiments that have their own Haxe defaults:

- Enabled by default:
  - `RAYZOR_HAXE_MATMUL=1`
  - `RAYZOR_HAXE_FLASH=1`
  - `RAYZOR_KV_Q8=1`
  - `RAYZOR_REQUANT_LM_HEAD=1`
  - `RAYZOR_PREFILL_LAST_LOGITS=1`
  - `--release`
  - `--llvm`
  - `--tier-promotion true`
  - `--tier-start-interpreted false`
  - thresholds `1/30/5/max`
- Not force-exported anymore:
  - `RAYZOR_HAXE_FUSED_MATMUL`
  - `RAYZOR_HAXE_FLASH_POOL`
  - `RAYZOR_STDOUT_FLUSH_MS`

If those env vars are unset, diagnostics print `auto` and the owning code
chooses.

## Current measurements and interpretation

Use these as direction, not final truth. The user had background jobs during
some runs, and profile/stat modes are invasive.

- Current signed-flash recovery restored correctness on the long Voronoi
  prompt.
- Silent, unprofiled long-prompt runs recovered into roughly the low-70 tok/s
  band on Mac in one clean run.
- Visible streamed output redirected to a file was still around low-60 tok/s.
  That means string/print streaming remains a real cost.
- `--stats` can drop throughput by a large amount. Do not compare `STATS=1`
  with `STATS=0`.
- Pre-NUC old source/toolchain did not reproduce the historical 80+ tok/s band
  under the same current conditions. A blind revert is not the answer.

## Known-bad or risky paths

- `RAYZOR_HAXE_FLASH_SHIFTED_Q=1`: opt-in only; can corrupt generation.
- `RAYZOR_HAXE_FUSED_MATMUL=1`: experimental; previously regressed full
  llama-chat.
- Speculative decode: implemented enough to run, but slower on CPU in recent
  measurements.
- Async Channel streaming: previous attempts failed because `Channel<String>`
  and static globals interacted badly with JIT/thread state. If retried, make a
  concrete `StreamMsg` class with `@:derive([Send])`, pass the channel through
  captured object state, and avoid static JIT globals.
- Profiling/stat modes: useful for fractions, not throughput baselines.
- Small-token prompts: useful for correctness smoke only, not performance.

## What to do next

1. Rebuild the CLI and bundle cleanly.
2. Run one long-prompt silent baseline with stats off.
3. Run one long-prompt visible-stream baseline with stats off.
4. If visible is materially slower, optimize streaming/printing. Do not touch
   flash/matmul first.
5. If TTFT stays high, inspect prefill morsel/worker behavior with a focused
   profile. Do not use decode tok/s as the proxy.
6. Re-test on the NUC only after resyncing and rebuilding the CLI there.
7. Keep commits grouped: codegen/runtime safety, Nue hot-path changes,
   benchmark/scripts, and docs.

Suggested local command:

```bash
cd nue/examples/llama-chat
BUILD=1 STATS=0 RAYZOR_LLAMA_SILENT_STREAM=1 MAX_TOKENS=732 ./run_bundle.sh
STATS=0 RAYZOR_LLAMA_SILENT_STREAM=0 MAX_TOKENS=732 ./run_bundle.sh
```

Suggested NUC command shape:

```bash
cd ~/rayzor/nue/examples/llama-chat
STATS=0 RAYZOR_LLAMA_SILENT_STREAM=1 MAX_TOKENS=732 ./run_bundle.sh
```

## Summary for Claude

Do not assume the regression is unrecoverable, and do not switch back to Rust
FFI to hide it. The immediate recovery path is:

- keep Haxe flash on,
- keep signed flash as default,
- keep fused matmul opt-in,
- keep stats/profile out of performance baselines,
- stop scripts from constraining Mac defaults,
- fix visible streaming cost separately,
- and treat LLVM missing-value errors as correctness wins, not regressions.

