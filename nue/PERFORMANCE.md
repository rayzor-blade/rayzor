# nue Performance — Pure-Haxe Kernel Parity

Companion to [ARCHITECTURE.md](ARCHITECTURE.md#performance-levers--the-ledger).
That ledger tracks the runtime/FFI levers; **this doc tracks the pure-Haxe
kernels and how close they are to the Rust kernels they must replace.**

**Standing direction:** every Tensor/QTensor kernel moves to pure Haxe and must
reach Rust parity on the Rayzor runtime. Rayzor is a *Haxe* runtime — Nue is its
proof. A measured Haxe-vs-Rust gap is a **work item against the kernel, the
pool, or the compiler**, never a reason to route that scheme back to Rust.

FFI is reserved for genuine **platform APIs** — Apple AMX via Accelerate
(`RZT_AMX_PREFILL`, prefill only, batch ≥ 16), the CoreML prefill graph
(`nue.engine.PrefillGraph`, ~15× per-op Q4 prefill), the BNNS Graph encoder
(`nue.engine.BertGraph`), and x86 VNNI (`NUE_INT8=1` → `vpdpbusd`, 2.15× on BERT
projections). Those are hardware/OS surfaces, not kernels Haxe could replace, so
using them is not a retreat from the direction. See ARCHITECTURE.md
§Platform Acceleration. **They are all opt-in or platform-gated and fall back to
the portable Haxe path**, so a parity measurement must state whether they were
active — an AMX-assisted prefill number is not a pure-Haxe number.

---

## Parity scorecard

Qwen2.5-0.5B, M1 Pro, `bench_server.sh`, **interleaved** arms, medians,
census-verified `total ffi=0 (PURE HAXE)`. Last measured 2026-07-27.

| Model | pure Haxe | Rust kernel | verdict |
|---|---|---|---|
| **Q6_K** (k-quant + Q8_0) | **125.69** | 106.30 | **Haxe wins +18.2%** (quiet machine, n=5 vs 2) |
| **Q5_0 → INT8** | 113.82 | 119.37 | **Haxe −4.6%** (non-overlapping) — narrowing |

Parity is **met and beaten for k-quant/Q8_0**; INT8 is **within 5%** and closing.

**Caveat on the INT8 number.** It read −18.9% (94.53 vs 116.52) earlier the same
day. The code change between the two readings (widening row-wise fusion to Q8_0
triples) should NOT affect an all-INT8 model, so most of that swing is the Haxe
arm's own run-to-run instability, not a fix. What *is* new and reproducible: the
Haxe values tightened from 84.4–96.4 to 110.5–114.5. Treat −4.6% as the current
best estimate, not a settled figure, until the variance (open item 2) is
understood.

---

## Landed (measured wins)

| Lever | Gain | Where |
|---|---|---|
| SpinPool const-reassign box leak fix | ~1GB/request → flat RSS | `SpinPool.hx` (e6e67eb7) |
| All-INT8 triple fusion (share activation quantise) | +21.3% / +25.0% Q5_0 | `Q4Matmul.matmulFused` (a1dc2c87) |
| Fused single-dispatch banding (joined row space) | +13.2%; dispatches −49.7% | `Q4Matmul.int8BandedFused` (584df55f) |

### The leak (worth remembering as a bug *class*)

`if (tightUs > 500) tightUs = 500;` — a conditional reassign of a local to a
**constant** — boxes that constant via `haxe_box_int_ptr` (48 bytes, `into_raw`,
never freed). In SpinPool's tight-spin loop that was **9M+ leaked boxes per
request (~1GB)**. Fix: ternary into a **fresh local**, never `if (c) v = <const>`.
Int variant of the same defect as `bugs_float_conditional_reassign_boxes`.
**Audit hot loops for this pattern.**

---

## Settled questions — do NOT retry these

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Copy Rust `dot_i8_i8`'s loop shape (2 acc / step 32) | **WRONG — 5.6% slower** | 90.25 vs our 95.30, non-overlapping (3612e0da) |
| Activation quantise is worth optimising | **No — 0.1% of band** | `quant_ms=3.4` vs `band_ms=2151.7` |
| `SIMD16i8_extract` select-chain is the INT8 cost | **Stale** — already worked around by `scaleI8`; `int8DotRow` uses no dynamic-lane get |
| `qw.scheme()` FFI traffic is significant | **No — ~0.02% of frame** | 2 calls/matmul, ~2µs/token, no allocation |
| Fusion's win comes from saving quantise work | **No** — quantise is 0.1%; the win is per-call/dispatch overhead |

---

## Open, with evidence

1. **INT8 is −18.9% vs Rust.** The deficit is *outside* the dot (our shape is
   proven better) and *outside* dispatch count (halved). Our k-quant band beats
   its Rust counterpart, so it is specific to the INT8 band's surroundings.
   Unexamined: weight streaming / cache blocking over rows×K,
   `QTensor.fusedQkvIntoArr`'s single-pass traversal, per-row scale-load + f32
   store tail.
2. **The Haxe pool path is UNSTABLE run-to-run — now the top item.** First seen
   on INT8 (84.4 / 94.5 / 96.4 while the Rust arm held 116.1 / 116.5 / 116.7,
   σ≈0.3), but it is **not INT8-specific**: a Q6_K stream A/B swung
   97.7–123.5 (±12%) within one arm on identical config. It tracks the Haxe
   pool, on any model.

   This is the **dominant source of uncertainty in every parity claim**: the
   same INT8 config read −18.9% in the morning and −4.6% in the evening.

   **It is REAL, not a measurement artifact.** I briefly attributed it to my own
   session's load (the box carried load ~4.7, WindowServer 28%, VS Code 26%,
   Chrome 25%) — that amplifies it, but a clean back-to-back pair on a quiet
   machine settles the question:

   Quiet-machine Q6_K observations — Haxe n=5: **103.30, 117.46, 125.69,
   125.85, 126.20**; Rust n=2: 105.22, 107.38 (spread 2.16).

   **The shape is an OCCASIONAL STALL, not gaussian jitter.** Four of five
   cluster at 117.5–126.2 (spread 8.7) with a single outlier at 103.3. Even
   that worst run is only −2.8% against Rust's median, while the Haxe median is
   **+18.2%** ahead. So this costs a rare ~18% drop, not a broad slowdown.

   **Prime suspect: the adaptive tight-spin bound landed 2026-07** (in the
   box-leak fix). It sizes the spin window from the pool's EWMA dispatch gap and
   parks after ~4x it, so a skewed EWMA — e.g. after the inter-request pause —
   parks workers early and pays wake latency on the next dispatch. That produces
   exactly one slow run among fast ones. **`NUE_POOL_ADAPTIVE=0` reverts to the
   iteration-only bound for A/B** (output bit-identical; the two are
   indistinguishable under contention, so this needs a quiet machine and several
   runs since the stall is ~1 in 5). Secondary suspect: the all-P-core spinning
   pool vs Rust's scoped per-call threads — test `cooperative`/`latency`.

   **Benchmark hygiene must therefore include `uptime` / top-CPU, not just
   "no lingering rayzor processes".** Numbers taken under GUI load are not
   comparable to numbers taken on a quiet box, and the all-P-core spin makes
   this framework unusually sensitive to it. Before treating variance as a code
   defect, re-measure quiet.
3. **Prefill is 1.9× slower than Rust** (0.077s vs 0.041s per 150-token run).
   Small on short prompts (~5% of wall), dominant on long ones — i.e. the server
   workload.
4. **Sampler costs 1.26 ms/token (12-13% of wall)**, identical in both arms and
   independent of quant scheme — a pure-Haxe win available regardless of matmul
   work (151,936-vocab logits + repetition penalty + no-repeat-8gram).

---

## Benchmarking rules (learned expensively)

- **INTERLEAVE THE ARMS.** This machine drifts up to **17% between batches
  minutes apart** (Q6_K read 113.64, 97.23, then 104.18 for the *same* config).
  Sequential batch comparisons produce confident, wrong conclusions — a "+11%"
  became −2.3% once interleaved. Alternate ON/OFF/ON/OFF, take medians, and
  **report whether the ranges overlap**. A non-overlapping gap is the only claim
  worth making.
- **Verify the kernel actually ran.** `NUE_DUMP_Q4_GATES=1` must print a
  per-scheme census line (`INT8 haxe=N ffi=0`). `total ffi=0` alone is a false
  positive — it also prints when nothing was counted.
- **Streaming is not a confound.** Measured interleaved on Q6_K: `STREAM=1`
  106.59 vs `STREAM=0` 102.59 — streaming measured *faster*, ranges overlapping,
  i.e. no real difference. Silencing the stream does not flatter a number; it
  just removes a variable.
- **Never compare profiled and unprofiled runs** (`NUE_PROFILE_DECODE` costs
  2-3 tok/s), and never sample memory during a timed run — a `vmmap` poller
  stole enough CPU to make a 117 tok/s config read 33.
- **Scrub `.rayzor` caches after editing imported modules or haxe-std**
  (`find . -name .rayzor -type d -exec rm -rf {} +`); `--no-cache` does not
  clear the BLADE cache.
- **The hot INT8 band cannot be disassembled by address.** It is promoted to
  LLVM, and `RAYZOR_DUMP_JIT_MAP` records only the Cranelift backend
  ("backend 0 … 0 defined functions (skip)"), so `int8Banded`/`int8DotRow` are
  absent while colder `q8Banded`/`runBanded` appear. Settle codegen questions by
  A/B measurement instead.

---

## Gates and defaults

Code defaults. The shipped runners (`run_bundle.sh`, `bench_server.sh`) export
`NUE_MATMUL=1 NUE_FLASH=1 NUE_KV_Q8=1 NUE_REQUANT_LM_HEAD=1
NUE_PREFILL_LAST_LOGITS=1` — that is the intended product configuration.

| Gate | Default | Effect |
|---|---|---|
| `NUE_MATMUL` | off in code, **=1 in every shipped runner** | routes quantised Linear through the pure-Haxe kernels |
| `NUE_HAXE_INT8` / `NUE_HAXE_Q8_0` | **on** | per-scheme Haxe band kernels |
| `NUE_FUSED_ROWWISE` | **on** (all-INT8 triples only) | share one activation quantise across a triple |
| `NUE_FUSED_DISPATCH` | **on** | band a fused triple's joined row space in ONE pool dispatch |
| `NUE_FUSED_MATMUL` | **off** — stale default, see below | k-quant row-space fusion (`runBandedFused`) |
| `NUE_FLASH` / `NUE_FLASH_POOL` | off in code / on | pure-Haxe flash decode; pooled kv-head bands |
| `NUE_POOL_SPINS`, `NUE_MATMUL_WORKERS`, `NUE_POOL_PROFILE` | platform defaults | pool tuning; see note below |

**`NUE_FUSED_MATMUL` is off on stale evidence.** Its comment claims the fused
path "regressed on macOS", but that verdict predates the SpinPool box-leak fix,
the interleaved methodology, and the 2026-07 finding that halving dispatches is
worth +13.2%. Re-measured interleaved: **Q6_K +3.1%** (116.95 vs 113.41) and
**Q4_K_M +1.8%** (111.98 vs 109.96) — positive on both, never negative, but the
ranges still overlap so it is not yet *established*. The "regressed" claim is
**refuted**; flipping the default needs a few more pairs, not a new argument.

Related: there are now three overlapping fusion gates —
`NUE_FUSED_MATMUL` (k-quant row-space), `NUE_FUSED_ROWWISE` (shared activation
quantise for row-wise triples) and `NUE_FUSED_DISPATCH` (single pool dispatch
over the joined row space). They should be consolidated behind one policy.

**Pool policy is workload-dependent** — the two arms want opposite settings.
When the pool is the compute engine (Haxe kernels on), a *lower* spin budget
helps (+4%) and `cooperative` (n−1 workers) *hurts* (−5.5%). When the pool is
idle (Rust kernels), worker count is the lever instead. A single global default
cannot serve both.

### Diagnostics

| Var | Purpose |
|---|---|
| `NUE_DUMP_Q4_GATES=1` | per-scheme kernel census (`haxe=N ffi=N`) |
| `NUE_PROFILE_POOL=1` | `band_ms` / `quant_ms` / `dispatches` |
| `NUE_PROFILE_DECODE=1` | per-phase decode timing (fwd / sample / prefill) |
| `RZT_LEAK_STATS=1` | live `tensor_net_live` vs `rss` — splits tensor from non-tensor growth |
| `RAYZOR_BOX_TRACE=1` | counts `haxe_box_int_ptr` calls **and logs the boxed value** — a boxed constant names its own source line; this is what found the SpinPool leak after backtraces, CLIF, LLVM IR and MIR all failed |
