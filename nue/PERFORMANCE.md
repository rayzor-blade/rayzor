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
| **Q6_K** (k-quant + Q8_0) | **135.63** | 106.30 | **best recorded** — darmie's bench, 2026-07-28 |
| Q6_K (previous) | 125.69 | 106.30 | Haxe +18.2% (quiet machine, n=5 vs 2) |
| **Q5_0 → INT8** | 113.82 | 119.37 | **Haxe −4.6%** (non-overlapping) — narrowing |

Parity is **met and beaten for k-quant/Q8_0**; INT8 is **within 5%** and closing.

**Q6_K reached 135.63 median on 2026-07-28** (`req_tps` 135.7/135.1/136.1 and
134.9/141.4/134.8 across two steady subprocesses; a third read 117.46 with
`ready=7.09s` — a cold first subprocess whose own three requests climb
110.1 -> 116.5 -> 126.9, so the reported `stddev=8.90 / drift=19.51` is entirely
that cold run). Candidate contributors, NOT attributed: the SpinPool adaptive
tight-spin floor fix (which targeted exactly this low tail) and the O(n log n)
tokenizer (per-request overhead; `ready` fell 7.09 -> 5.82 s).

**Caveat on same-day re-checks.** An interleaved Haxe-vs-Rust A/B run the same
afternoon gave Haxe 105.00 vs Rust 96.70 (+8.6%, n=9 each, ranges OVERLAPPING) —
same direction, weaker separation, and absolute numbers ~25% below the figures
above. That box had been benchmarking all day with the volume at 97-100% and
swap peaking at 21.5 GB. **Machine state moves the absolute number more than any
kernel change measured in this document**; only compare arms measured in the
same interleaved window, and prefer a quiet box for headline figures.

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
| AMX prefill threshold 16 → 128 (measured crossover) | short-prompt prefill −7%; 16-tok case −61% | `Q4Matmul.amxMinBatch()` |

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
2. **Decode stall — FIXED 2026-07-27.** The Haxe pool showed a rare ~18% drop
   (4 of 5 runs at 117.5–126.2, one at 103.3). Cause: the adaptive tight-spin
   window landed earlier the same day could END the spin BEFORE the iteration
   budget, parking a worker early and paying wake latency on the next dispatch.
   Fix: the measured dispatch-gap window is a **floor**, never a cutoff — it can
   only EXTEND spinning, which is what cross-machine calibration actually needs.

   Verified, 15 per-request samples per arm, interleaved with cooldowns:

   | adaptive spin | min | median | spread | >10% below median |
   |---|---|---|---|---|
   | cutoff (before) | 114.9 | 128.6 | 17.2 | **1/15** |
   | **floor (after)** | **122.0** | 127.9 | **9.7** | **0/15** |
   | iteration-only reference | 124.2 | 127.7 | 5.3 | 0/15 |

   Median is unchanged throughout — this was always a TAIL defect, which is why
   comparing medians would have missed it. `NUE_POOL_ADAPTIVE=0` still selects
   the plain iteration bound. Residual: iteration-only remains slightly tighter
   (spread 5.3 vs 9.7); re-check on a quiet box before chasing it.

3. **Prefill is 1.9× slower than Rust** (0.077s vs 0.041s per 150-token run).
   Small on short prompts (~5% of wall), dominant on long ones — i.e. the server
   workload.
5. **The BPE tokenizer is O(n^2) — it is the long-context wall, not the model.**
   `encodeBPE` rescans every adjacent pair to find the best merge and then
   REBUILDS the whole `pieces` array, once per merge: O(n) x O(n) merges, plus
   an allocation per iteration. Measured (Qwen Q6_K, `tokenize_s`):

   | prompt tokens | tokenize_s | vs previous |
   |---|---|---|
   | 347 | 0.282 | — |
   | 706 (2.03x) | 1.197 | **4.24x** |
   | 1402 (2.00x) | 4.866 | **4.07x** |
   | 2865 (2.04x) | 19.816 | **4.07x** |

   Doubling the prompt quadruples tokenisation. Extrapolated: ~160 s at 8k and
   **~43 MINUTES at Mistral's full 32k context**, before a single weight is
   touched. On the 2628-token Llama-1B run it was already 20.0 s against 52.4 s
   of prefill — 28% of time-to-first-token.

   Fix is standard and self-contained: track pair ranks in a heap and merge
   in place over a linked list instead of rescanning + rebuilding. Pure Haxe,
   no FFI, no numerics change.

4. **Sampler costs 1.26 ms/token (12-13% of wall)**, identical in both arms and
   independent of quant scheme — a pure-Haxe win available regardless of matmul
   work (151,936-vocab logits + repetition penalty + no-repeat-8gram).

---

### AMX prefill: the gate was wrong, not the platform

`AMX_MIN_BATCH` was 16 — 8x too low. Interleaved arms, cooldown between every
run, prefill timed directly (`prefill_fwd_s`, Qwen Q5_K_M):

| prompt tokens | AMX on | AMX off | delta |
|---|---|---|---|
| 16 | 0.0926 | 0.0574 | **+61.2%** AMX slower |
| 39 | 0.1639 | 0.1565 | +4.8% slower |
| 69 | 0.2553 | 0.2368 | +7.8% slower |
| **129** | 0.5096 | 0.5124 | **−0.5% (crossover)** |
| 249 | 1.2108 | 1.2870 | −5.9% faster |
| 429 | 2.8065 | 2.9358 | −4.4% faster |

Set to **128** (`NUE_AMX_MIN_BATCH` overrides — other silicon will differ).
Verified after: a 69-token prompt now reports `platform=0` and prefills in
0.2384 (was 0.2553); a 429-token prompt still reports `platform=12` and is
unchanged.

**The lesson generalises: a platform path is never "remove it to be pure" — it
is "is it gated correctly?"** Decode never takes this path at all, because
decode is batch=1 and bandwidth-bound.

### …and the gate is STILL wrong, because batch size is the wrong variable

Measured on Mistral-7B-Q4_K_M (2026-07-27), which is the first model where the
AMX gate actually engaged (`platform=192` = 32 layers x 6 weights):

| | |
|---|---|
| one-time f16 dequant | **192 weights, 12.63 s** (`RZT_AMX_DEBUG=1` prints each) |
| RSS cost | **+1.59 GB** (6.09 vs 4.50) |
| prefill with AMX | 18.88 s |
| **prefill minus one-time dequant** | **6.25 s** |
| prefill AMX off | **9.02 s** |

So the AMX GEMM is **~31% FASTER**; a naive interleaved A/B reads **+86.2%
slower** because a one-shot run builds the cache inside the prefill it is
timing. **A lever with a one-time cost cannot be judged by a one-shot run.**

`AMX_MIN_BATCH` gates on batch size, but batch size does not predict whether a
weight's f16 copy gets REUSED — which is the actual economics.

### The ceiling: every Apple path materialises f16, undoing the quantisation

`cached_f16_weight` leaks one f16 copy per routed weight by design (BNNS/AMX
consume f16 only — `BNNSMatMul` int8 was probed and REJECTED, "the public
matmul is float-only"). Warming all of a 7B's weights therefore asks for a
second, fatter copy of the model:

    layer params 7.78 B -> f16 14.5 GB  +  4.07 GB quantised  =  18.6 GB  vs 16 GB

Attempted and REVERTED (nothing committed): warming via a batched `forwardIds`
took **491 s**; a lean per-weight prewarm did the same job in **16 s**, but the
box then sat at 12.4 GB with an idle CPU — swapping, not computing.

The same wall bounds the CoreML prefill graph: fp16 artifacts are ~2.0 GB per
bucket at 1B and **14.8 GB at 7B**. BNNSGraph does not escape it either — it
executes the *same* mlmodelc and computes in fp16 (it is a shipping engine for
BERT at ~400 sent/s, cos 0.9999628, but was never wired to LLM prefill).

**One wall, not three.** The escape is an artifact that stays compressed:
4-bit palettized would be 3.6 GB/bucket at 7B and is readable by CoreML *and*
BNNSGraph without touching engine code.

Measured (Llama-1B s128, fp16 artifact = 1856 MB, baseline cos 0.999141):

| config | size | ratio | `out` cos | k min | v min |
|---|---|---|---|---|---|
| fp16 (baseline) | 1856 MB | 1.000 | 0.9991 | — | — |
| 4-bit, per-tensor | 464 MB | 0.250 | **0.181** | 0.9860 | 0.9764 |
| 4-bit, grouped ch. (32) | 465 MB | 0.250 | **0.754** | 0.9914 | 0.9825 |
| **6-bit, grouped ch. (32)** | **698 MB** | **0.376** | **0.9973** | 0.9995 | 0.9989 |

The 4x compression is exactly as predicted, and granularity helps a lot
(0.181 -> 0.754) at no size cost — but **4-bit fails the coherence gate either
way**. 0.75 on the final hidden state is not shippable for a decoder.

**6-bit grouped is the configuration that fits AND nearly holds.** Projected to
7B: 5.4 GB/bucket + 4.07 GB quantised = 9.5 GB, inside 16 GB (fp16 would be
18.6 GB). 0.9973 sits just under nue's customary 0.999 accept gate.

END-TO-END (the gate that actually matters — same prompt, temp=0, artifact
swapped under the runner): both arms produce fluent, factually correct text and
**agree for ~25 tokens, then diverge**. At greedy decoding that means the argmax
changed, so 6-bit is **NOT a drop-in equivalent** of fp16 — it is a
different-but-valid completion. Whether that is acceptable is a product call,
not a measurement one.

Do NOT read decode tok/s from that A/B (80.98 vs 64.80): the prefill graph runs
PREFILL only, `ttft` was 0.0910 vs 0.0928 (identical), and decode is the same
pure-CPU path in both arms. The gap is machine noise.

### The graph's value is dominated by BUCKET FIT, not by compression

Artifacts are FIXED-shape, so a prompt is padded up to the next bucket and the
graph computes the full bucket. Attention is O(S^2), so the padding is not a
linear tax. Llama-1B, `NUE_PREFILL=on` vs `off`, 3 pairs each:

| prompt | bucket | graph on | graph off | verdict |
|---|---|---|---|---|
| 115 tok | s128 (11% pad) | **0.0945** | 1.7157 | **18.2x FASTER** |
| 193 tok | s512 (62% pad) | 2.650 | 1.966 | ~35% slower |

**18x when it fits; a liability when it does not.** 128 rows cost 0.0945 s while
512 rows cost 2.65 s — 4x the rows for ~28x the time. So bucket COVERAGE is the
policy that matters, and it is a placement decision (NueGraph Stage 4), not a
compression one. A first-run outlier (7.78 s) is one-time artifact init — do not
include it in a median.

This is also what makes a 7B artifact worth building despite the cost: Mistral's
prefill is **76% of wall** (12.54 s of 16.52 s), so an 18x on the aligned case is
transformative — far larger than any kernel win measured in this document.

**NOT blocked after all — the full volume was self-inflicted.** Authoring needs
~14.5 GB for the fp16 intermediate plus ~5.4 GB for the 6-bit copy (~20 GB), and
the attempt was abandoned with the volume at 97-100%. That capacity was **286.5
GB of CoreML compiled-bundle cache generated by the palettization sweeps
themselves** — every `MLModel(...)` load leaves a copy of the compiled model in
`~/Library/Caches/org.python.python/com.apple.e5rt.e5bundlecache`, and the
sweeps loaded 1.8 GB models hundreds of times (474 bundles). Clearing only the
session's own entries returned the volume to 320 GB free / 64%.

**RETRIED AND MEASURED (2026-07-28) — the graph LOSES at 7B.** Authored the
fp16 s128 bucket for Mistral-7B (13 GB, ~11 min, coherence `hidden=0.999847`
BETTER than the 1B's 0.999141), palettized it to 6-bit grouped (4.88 GB, ratio
0.375 — matching the 1B's 0.376, ~2 h at 4.9->31.5 s/op as the wide FFN tensors
land), compiled and installed both.

| 7B arm | gen tok/s | prefill | load | wall | end-to-end tok/s | RSS |
|---|---|---|---|---|---|---|
| 6-bit graph | 1.07 | 45.64 s | 115.9 s | 176.6 s | **0.34** | 4.49 GB |
| fp16 graph | — | 48.86 / 48.68 s | 238 / 118 s | — | — | — |
| **CPU prefill** | **1.84** | **16.69 s** | **10.9 s** | **47.7 s** | **1.26** | **4.20 GB** |

**CPU wins on every axis — 3.7x end-to-end, and less memory.** Discounting load
entirely (fair for a server) CPU prefill is still 2.7x faster. There is no
regime at 7B where the graph pays.

**Palettization is NOT the cause: fp16 measured marginally WORSE than 6-bit**
(48.7 vs 44.3 s), so 6-bit is a mild win on both size and speed. The graph
itself does not scale — 18.2x FASTER at 1B, ~3-4x SLOWER at 7B.

**Revised conclusion: the CoreML prefill graph is a SMALL-MODEL lever (<=1-2B).**
Bucket policy still matters, but only for models where the graph wins at all.
Direct-palettized authoring is no longer worth building for 7B — it would make
a losing artifact cheaper to produce.

(Superseded prediction, kept deliberately: this section previously argued a 7B
artifact "should be retried" because Mistral's prefill is 76% of wall and an
aligned 18x would be transformative. The 18x did not survive the scale-up.)

**Original note, now historical:** Baking
palettized weights directly (`constexpr_lut_to_dense`, LUT computed at author
time) is still the better design — ~5.4 GB instead of ~20 GB, and no fp16
intermediate — but it is now an optimisation, not a prerequisite.

**Operational rule: the e5rt bundle cache grows without bound across CoreML
work.** Check it before and after any authoring/palettization session; it is
pure cache and safe to clear, at the cost of a one-time recompile.

**The failure shape is the reusable lesson: per-layer k/v stayed at ~0.99 in
BOTH configs while the final hidden collapsed**, because error compounds through
16 residual-stream layers. Judging palettization on per-layer cosines alone
would have passed a config that destroys the output. Gate on the FINAL hidden.

**Measurement cost note:** ~4 min per config, single-threaded. Do NOT set
`num_kmeans_workers>1` — children re-run the module via `runpy.run_path` and
trip multiprocessing's import guard regardless of `__main__` protection; it
wasted 18 min at ~5 s of CPU and blew swap from 11.5 to 21.5 GB.
`cputime` vs `elapsed` is the stall test: 5 s/18 min = blocked,
53 s/60 s = working.

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
- **A zero counter can mean "unobserved", not "did not happen".** The
  `[nue-plan] fusion:` line read `fused=0 split=0` on every k-quant model,
  which looks exactly like "fusion declined" but actually meant the k-quant
  banded path had **no instrumentation at all** — only the row-wise path
  incremented. The `[nue-plan] sites:` line now closes this: it counts call
  sites *before* any gate, so `sites − fused − split` (`skipped-dispatcher`)
  separates **forfeited** from **declined**, and `haxe-matmul-off-at-site` /
  `unquantised` name the two reasons a site bails early. Before trusting a zero,
  confirm the path that would have incremented it is on the counted list.
- **Streaming is not a confound.** Measured interleaved on Q6_K: `STREAM=1`
  106.59 vs `STREAM=0` 102.59 — streaming measured *faster*, ranges overlapping,
  i.e. no real difference. Silencing the stream does not flatter a number; it
  just removes a variable.
- **Say whether the run was WARMED.** `NUE_DECODE_WARM` is default ON and warms
  both decode entry points at load; without it request 1 decodes partly
  pre-tier-promotion. Every number in this document is warmed. `NUE_DECODE_WARM=0`
  is a legitimate and *different* measurement — what a user's first request
  actually costs — and it has never been measured here.
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

Code defaults. **A plain `rayzor run` now takes the full pure-Haxe path — no
env flags required.** The runners still export them for explicitness.

`NUE_MATMUL`, `NUE_FLASH` and `NUE_KV_Q8` were flipped to default-ON on
2026-07-28. They had been opt-in while every shipped runner set them, so the
default path silently used the Rust XTQ kernels and an F32 KV cache. Measured
on Mistral-7B-Q4_K_M with NO flags, before vs after:

| | max RSS | tok/s |
|---|---|---|
| before (opt-in) | 7.58 GB | 0.115 |
| **after (default-on)** | **4.59 GB** | **0.422** |

**-2.99 GB and 3.7x on the path users actually take.** Output is unchanged
(bit-identical across a 60-token Qwen generation for the FLASH/KV_Q8 flip), and
the census now reads `haxe_matmul=on(observed)` + `PURE HAXE` with no flags on
Qwen, Llama and Mistral. Isolated contributions: `NUE_MATMUL` -2.04 GB / 5.2x,
`NUE_FLASH`+`NUE_KV_Q8` a further -1.02 GB.

**Two gates read `NUE_FLASH`** — `LlamaArch` (build the cache as guest-owned
Q8) and `FlashDecode.enabled()` (use the kernel). They must be flipped
together; opposite defaults would build a Q8 cache nothing reads.

| Gate | Default | Effect |
|---|---|---|
| `NUE_MATMUL` | **on** (`=0` opts out) | routes quantised Linear through the pure-Haxe kernels |
| `NUE_HAXE_INT8` / `NUE_HAXE_Q8_0` | **on** | per-scheme Haxe band kernels |
| `NUE_FUSED_ROWWISE` | **on** (all-INT8 triples only) | share one activation quantise across a triple |
| `NUE_FUSED_DISPATCH` | **on** | band a fused triple's joined row space in ONE pool dispatch |
| `NUE_FUSED_MATMUL` | **off** — stale default, see below | k-quant row-space fusion (`runBandedFused`) |
| `NUE_FLASH` / `NUE_KV_Q8` | **on** (`=0` opts out) | pure-Haxe flash decode + guest-owned Q8 KV cache |
| `NUE_FLASH_POOL` | **on** | pooled kv-head bands |
| `NUE_POOL_SPINS`, `NUE_MATMUL_WORKERS`, `NUE_POOL_PROFILE` | platform defaults | pool tuning; see note below |
| `NUE_DECODE_WARM` | **on** | warms both decode entry points at load, so request 1 is not measured pre-tier-promotion |
| `NUE_AMX_MIN_BATCH` | **128** | measured AMX prefill crossover (see above) |

**`NUE_PREFILL_WARM` IS A DEAD FLAG — nothing reads it.** It was renamed on
2026-07-22 (04162bd1): `warmPrefill()` was gated on the CoreML graph, so a
pure-CPU server could not warm decode at all, and since the bundle
**tier-promotes by call count** an unwarmed first request decoded largely
pre-promotion (~42 vs ~90 steady tok/s). It became `warm()`, ungated by the
graph, behind `NUE_DECODE_WARM` (default ON). That commit measured pure-CPU
median 54 -> 81 tok/s and first request 42 -> 66.

Both of us kept passing `NUE_PREFILL_WARM=1` on bench command lines long after
it stopped existing. The runs *were* warm — because `NUE_DECODE_WARM` defaults
on, visible as the `[warm] model warmed in ...s` line — but not for the reason
the command line implied. Drop it from invocations.

**`NUE_FUSED_MATMUL` is off on stale evidence.** Its comment claims the fused
path "regressed on macOS", but that verdict predates the SpinPool box-leak fix,
the interleaved methodology, and the 2026-07 finding that halving dispatches is
worth +13.2%. Re-measured interleaved: **Q6_K +3.1%** (116.95 vs 113.41) and
**Q4_K_M +1.8%** (111.98 vs 109.96) — positive on both, never negative, but the
ranges still overlap so it is not yet *established*. The "regressed" claim is
**refuted**; flipping the default needs a few more pairs, not a new argument.

**2026-07-27 — the extra pairs say do NOT flip it yet.** The gate is a no-op on
any model with row-wise weights (`canFuseRowwise` already returns true via
`NUE_FUSED_ROWWISE`), so it only bites on **pure k-quant** models — the Llama
family. On Llama-3.2-1B-Q4_K_M, 5 interleaved pairs at 120 tokens:

| arm | samples (tok/s) | median | spread |
|---|---|---|---|
| fused ON | 88.4 · 59.9 · 84.2 · 85.9 · 66.4 | 84.17 | **28.5** |
| fused OFF | 83.3 · 83.7 · 82.7 · 82.0 · 80.2 | 82.70 | 3.5 |

Median **+1.8%, ranges overlap**. The ON arm carries two ~30% low outliers while
the OFF arm stays tight *in the same interleaved window*, so the variance is
arm-specific, not machine drift. Output is **bit-identical** between arms, so
this is purely a latency-tail question.

A 3-pair run of this same comparison read **+4.7% with non-overlapping ranges**
and was wrong. Three pairs is not enough for a default change; that is now the
bar. Characterise the tail before default-on — it resembles the decode-stall
signature (low outliers, not a shifted median) but survives the
[SpinPool floor fix](#) and lives on the single-dispatch joined-row-space path.

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
