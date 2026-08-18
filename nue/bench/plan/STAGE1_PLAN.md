# NueGraph Stage 1 — implementation plan

Produced by an 18-agent survey/design/adversarial-judge pass over the real code.
Every claim below was verified against a file:line by at least one agent, and every
judge-found fatal flaw is already folded in.

## Chosen design

NueGraph Stage 1 — build-time route plan, co-located in `nue.arch`, lowered to tri-state Int fields (synthesis of design #1's hoist + design #2's verify mode + design #3's phase/cache-kind discipline, with all judge-found fatal flaws fixed)

## Rationale

Design #1 (avg 5.83) is the base: hoist the per-token fusion decision into a build-time plan, lower it to a primitive field on the node, never make Q4Matmul read anything new. Grafted from #2: the `NUE_PLAN` VERIFY mode that compares the planner's decision against the live expression on every forward (a per-call equivalence proof in ONE binary, not a recompile A/B), and the refusal to make `amxPrefill`/`useHaxeQ8_0`/`useHaxeInt8`/`nrowBlock`/`shiftedQ`/`usePool` public. Grafted from #3: record the BUILT cache kind (`cache.useQ8H`/`useQ8`, not the request) and phase-key the dispatch, plus the append-only field rule.

Every judge fatal flaw is fixed, and I verified each one against the code rather than trusting the verdicts:

1. NULL DEREF ON THE EXAMPLES (real, verified). `grep "new GQAttention\|new SwiGLU"` returns examples/tiny-transformer/Main.hx:148,152 and examples/transformer-pieces/Main.hx:34,112 — instances built outside LlamaArch that immediately call `forward`. FIX: the lowered field is a plain `Int` with 0 = unplanned, and 0 means "evaluate live exactly as today". Those two examples need zero edits and keep today's behaviour. This is not a resolution fallback; it is the documented tri-state idiom (Linear.hx:65-70 "Zero-valued uninitialised is load-bearing").

2. `-1` SENTINEL (real). Verified: LlamaModel.hx:58 `prefillHandle:Int = 0`, LlamaBlock `_debugShapes:Int = 0`, Linear `_haxeMatmul:Int = 0` — there is no non-zero Int initialiser in nue. FIX: 0 = unplanned; states are 1=on, 2=off, 3=on+verify, 4=off+verify. A zeroed cross-module duplicate falls back to live evaluation instead of decoding to a plausible-but-wrong route. No bit masks, so no shared encoding to drift across three files.

3. `Null<nue-class>` FIELD (real). Verified: all 7 nullable instance fields in nue are rayzor-stdlib types (Linear.hx:39,40,45,50; Embedding.hx:34; LayerNorm.hx:20; LlamaModel.hx:37). Zero nue-defined classes in a `Null<>` field, against an open corruption record. FIX: no `LlamaModel.plan` field at all in Stage 1. The plan is a LOCAL in `LlamaArch.build()`, dumped there. Cost: `prefillHandle` is set later by the loader, so the dump prints `graph_prefill=deferred` — accepted and stated.

4. NEW PACKAGE / CROSS-MODULE FIELD READS (real). FIX: `NuePlan.hx` goes in `package nue.arch` alongside `LlamaArch.hx` (co-location rule; no new directory, no new package). LlamaArch does all the field reads (`cache.useQ8H` is already read at LlamaArch.hx:207) and passes primitives into `addNode`, so NuePlan reads no foreign field ever.

5. PREFILL ORACLE VACUOUS (real). Verified bench/eval/gate.sh:78 `ab_out="$( export "${AB_VAR?}"; run_eval ...)"` — same binary, one extra env var, and :67 ends `2>/dev/null`. `--ab NUE_PLAN=1` could never prove anything. FIX: run `bench/eval/Main.hx --dump` directly in the BEFORE tree and the AFTER tree and join the two TSVs; stderr captured.

6. PIN LIST HOLES (real). Added NUE_FLASH_SHIFTED_Q, NUE_REQUANT_Q6K, NUE_PREFILL, NUE_PREFILL_LAST_LOGITS, NUE_NROW_BLOCK, NUE_DUMP_BLOCK_SHAPES, NUE_PROFILE_DECODE_SPLIT, plus a hard scrub of stray RAYZOR_*/RZT_* aliases (Sys.hx:65-69 makes an unset NUE_* name silently inherit the alias).

7. `[nue-plan]` PREFIX COLLISION (real). Verified Q4Matmul.hx prints four `[nue-plan]` lines. New dump uses `[nue-graph]`.

8. COMMIT 1 SELF-CONTRADICTION (real). Verified Q4Matmul.hx:106-121: `useFusedMatmul()` prints `[q4-gate]` as a FIRST-READ side effect, and LlamaArch does not call it today (its only Q4Matmul calls are workerCount/poolSpins/poolRelax/poolProfiling/poolAdaptive at :131/:134). FIX: step 1 records only gate-call-free facts, so `[q4-gate]` provably does not move; the move is isolated into step 2 where it is the only expected delta.

9. INCREMENT 4 / hot-loop rewrite (design #3's `plan.forwardLayers`) is CUT from Stage 1 — it buys nothing when exactly one layer kind exists and it touches an aliased in-place residual (LlamaBlock.hx:99 `x.addInto(attnOut)`) in a compiler with a recorded loop-drop miscompile.

The route is computed in `buildBlock`, where `qProj/kProj/vProj/attn` are already locals — verified LlamaArch.hx:219-240. Only SwiGLU's three Linears are built inline at :234-238 and must be bound to locals first (per-layer weight promotion means Q6_K can appear on some layers, so the route is per-layer, not per-model). No signature on Q4Matmul changes; the file is not edited at all.

## Gate table (MUST be preserved exactly)

Every gate below keeps its OWN truthiness test and its OWN default. NOTHING is normalised — the policy object CALLS the existing accessor or moves the existing expression character-for-character. `Sys.getEnvOr(name, alias)` (compiler/haxe-std/Sys.hx:65-69) returns the primary, else the alias, else null.

PLAN-VISIBLE (owned or reported by NuePlan)
| var | alias | owner:line | test | default | pin |
|---|---|---|---|---|---|
| NUE_MATMUL | RAYZOR_HAXE_MATMUL | Linear.hx:85-92 | off iff "0"/"false" | ON | =1 |
| NUE_INT8 | RAYZOR_INT8 | Linear.hx:98-104 | on iff !null/!"0"/!""/!"false" | OFF | =0 |
| NUE_FUSED_MATMUL | RAYZOR_HAXE_FUSED_MATMUL | Q4Matmul.hx:106-113 | on iff !null/!"0"/!""/!"false" | **OFF (opt-in)** | =0 |
| NUE_FUSED_ROWWISE | RAYZOR_HAXE_FUSED_ROWWISE | Q4Matmul.hx:1165-1171 | off iff "0"/"false" | **ON** | =1 |
| NUE_FUSED_DISPATCH | RAYZOR_HAXE_FUSED_DISPATCH | Q4Matmul.hx:1182-1188 | off iff "0"/"false" | **ON** | =1 |
| NUE_AMX_MIN_BATCH | RZT_AMX_MIN_BATCH | Q4Matmul.hx:65-72 | parseFloat, >0 wins | 128 | =128 |
| NUE_KV_Q8 | RAYZOR_KV_Q8 | LlamaArch.hx:87-88 | off iff "0"/"false" | ON | =1 |
| NUE_FLASH | RAYZOR_HAXE_FLASH | LlamaArch.hx:95-105 AND FlashDecode.hx:36-42 | off iff "0"/"false" | ON | =1 |
| NUE_FLASH_BATCH | RAYZOR_HAXE_FLASH_BATCH | FlashDecode.hx:109-117 | numeric CAP, stored n+1 | uncapped (1<<30) | **=1073741824, NEVER =1** |
| NUE_REQUANT_LM_HEAD | RAYZOR_REQUANT_LM_HEAD | LlamaArch.hx:171-172 | on iff `!= "0"` (does NOT accept "false") | ON | =1 |
| NUE_REQUANT_Q6K | RAYZOR_REQUANT_Q6K | LlamaArch.hx:269 | on iff `== "1"` | OFF | =0 |
| NUE_POOL_PROFILE | RAYZOR_HAXE_POOL_PROFILE | Q4Matmul.hx:1081-1087 | string, not memoised | "throughput" | =throughput |
| NUE_MATMUL_WORKERS | RAYZOR_HAXE_MATMUL_WORKERS | Q4Matmul.hx:1092-1104 | parseFloat >0 wins | perfCoreCount() | =8 |
| NUE_POOL_SPINS | RAYZOR_HAXE_POOL_SPINS | Q4Matmul.hx:1106-1118 | parseFloat >0 wins | profile-derived (0 for throughput) | leave UNSET |
| NUE_POOL_RELAX | RAYZOR_HAXE_POOL_RELAX | Q4Matmul.hx:1128-1132 | tri-valued Int | **-1 = platform default** | leave UNSET |
| NUE_POOL_ADAPTIVE | RAYZOR_HAXE_POOL_ADAPTIVE | Q4Matmul.hx:1122-1125 | off iff "0"/"false" | ON | =1 |
| NUE_PREFILL | (none, bare getEnv) | GGUFLoader.hx:88-89 | off iff "off"/"0" | ON | **=off** |
| NUE_DECODE_WARM | RAYZOR_DECODE_WARM | GGUFLoader.hx:113 | off iff `== "0"` | ON | =1 |
| NUE_Q8_0_QUANT | RAYZOR_Q8_0_QUANT | GGUFLoader.hx:464-470 | off iff "0"/"false" | ON | =1 |
| NUE_FREE_GGUF_BYTES | RAYZOR_FREE_GGUF_BYTES | GGUFLoader.hx:202 | off iff `!= "0"` inverted | ON | =1 |
| NUE_PREFILL_LAST_LOGITS | RAYZOR_PREFILL_LAST_LOGITS | GenerationLoop.hx:418-421 | on iff truthy | **OFF in code, but run_bundle.sh:103 exports =1** | =0 (code default) |

KERNEL-OWNED — NOT adopted, NOT reported by NuePlan, accessor stays PRIVATE (verified: no `public` on any of these). Making them public is a declaration change on the two classes carrying the E0100 notes and is explicitly out of Stage 1.
| NUE_NROW_BLOCK | RAYZOR_NROW_BLOCK | Q4Matmul.hx:39-46 (private) | parseFloat >0 | 64 | =64 |
| RZT_AMX_PREFILL | RAYZOR_AMX_PREFILL | Q4Matmul.hx:78-86 (private) | off iff "0"/"false" **AND folds `Sys.systemName()=="Mac"`** | ON on Mac only | =1 |
| NUE_AMX_HAXE | RZT_AMX_HAXE | Q4Matmul.hx:97-104 (private) | off iff "0"/"false" | ON | =1 |
| NUE_HAXE_INT8 | RAYZOR_HAXE_INT8 | Q4Matmul.hx:161-167 (private) | off iff "0"/"false" | ON | =1 |
| NUE_HAXE_Q8_0 | RAYZOR_HAXE_Q8_0 | Q4Matmul.hx:149-155 (private) | off iff "0"/"false" | ON | =1 |
| NUE_FLASH_SHIFTED_Q | RAYZOR_HAXE_FLASH_SHIFTED_Q | FlashDecode.hx:44-57 (private) | on iff !"0"/!"false" (case-insens) | **OFF** — changes the query quantiser AND band fn (FlashDecode.hx:286-308) AND makes decodeBatch return null (:144) | =0 |
| NUE_FLASH_POOL | RAYZOR_HAXE_FLASH_POOL | FlashDecode.hx:60-71 (private) | off iff "0"/"false" | ON | =1 |

INSTRUMENT / DEBUG (pin identically in both arms)
| NUE_DUMP_Q4_GATES | RAYZOR_DUMP_Q4_GATES | Q4Matmul.hx:186-191 (+ a SECOND memo at :113-120) | truthy | OFF | =1 in arm A |
| NUE_PROFILE_POOL | RAYZOR_PROFILE_POOL | Q4Matmul.hx:1135-1138 | truthy | 0 | =1 in arm A (latched at pool ctor, LlamaArch.hx:131) |
| NUE_DUMP_BLOCK_SHAPES | RAYZOR_DUMP_BLOCK_SHAPES | LlamaBlock.hx:37-47 | **three-valued**: "0"/""/"false"/null→2 off; "trace"→3; else→1 | OFF | =0 |
| NUE_PROFILE_DECODE_SPLIT | RAYZOR_PROFILE_DECODE_SPLIT | Q4Matmul.hx:280-286 | truthy | OFF | =0 |
| NUE_PROFILE_ATTN | RAYZOR_PROFILE_ATTN | GQAttention.hx:67-73 | truthy | OFF | =0 |
| NUE_DUMP_TOPK | RAYZOR_DUMP_TOPK | LocalTempSampler.hx:91-97 | truthy | OFF | =0 (unusable as an oracle) |
| NUE_LLAMA_SILENT_STREAM + NUE_LLAMA_DUMP_OUTPUT | RAYZOR_* both | GenerationLoop.hx:423-429 | **two-variable AND** | emit text | =1 AND =1 together |

NEW GATE INTRODUCED BY THIS WORK — exactly one, and only as temporary scaffolding: `NUE_PLAN_VERIFY` / `RAYZOR_PLAN_VERIFY`, read ONCE in LlamaArch.build (never on the hot path), truthy test, default OFF, deleted in step 6.

## Baseline procedure

All commands from /Users/amaterasu/Vibranium/rayzor. Capture BEFORE at HEAD, before step 1.

--- 0. Build + hygiene ---
pkill -9 -f '[t]arget/release/rayzor'
CARGO_INCREMENTAL=0 cargo build --release -p rayzor --bin rayzor
find /Users/amaterasu/Vibranium/rayzor/nue -name .rayzor -type d -exec rm -rf {} + 2>/dev/null
# discard the first run after a build (page-cache artefact)

--- 1. The pin runner (create once, used by every arm) ---
Write /Users/amaterasu/Vibranium/rayzor/nue/bench/plan/pins.sh:

  #!/usr/bin/env bash
  # Scrub EVERY stray alias first: Sys.getEnvOr falls back to the alias when the
  # NUE_* name is unset, so a leftover RAYZOR_HAXE_* silently supplies a value.
  for v in $(env | grep -oE '^(RAYZOR|RZT|NUE)_[A-Z0-9_]*' ); do unset "$v"; done
  export NUE_MATMUL=1 NUE_INT8=0 NUE_HAXE_INT8=1 NUE_HAXE_Q8_0=1 \
    NUE_FUSED_MATMUL=0 NUE_FUSED_ROWWISE=1 NUE_FUSED_DISPATCH=1 \
    NUE_NROW_BLOCK=64 NUE_AMX_MIN_BATCH=128 RZT_AMX_PREFILL=1 NUE_AMX_HAXE=1 \
    NUE_FLASH=1 NUE_KV_Q8=1 NUE_FLASH_POOL=1 NUE_FLASH_SHIFTED_Q=0 \
    NUE_FLASH_BATCH=1073741824 \
    NUE_REQUANT_LM_HEAD=1 NUE_REQUANT_Q6K=0 NUE_Q8_0_QUANT=1 \
    NUE_DECODE_WARM=1 NUE_FREE_GGUF_BYTES=1 \
    NUE_PREFILL=off NUE_PREFILL_LAST_LOGITS=0 \
    NUE_POOL_PROFILE=throughput NUE_MATMUL_WORKERS=8 NUE_POOL_ADAPTIVE=1 \
    NUE_DUMP_BLOCK_SHAPES=0 NUE_PROFILE_DECODE_SPLIT=0 NUE_PROFILE_ATTN=0 \
    NUE_DUMP_TOPK=0
  # NUE_POOL_SPINS / NUE_POOL_RELAX deliberately LEFT UNSET (platform defaults)
  exec "$@"

--- 2. ARM A — text + dispatch counts (llama-chat, source path, NOT bench.sh) ---
bench.sh with BUNDLE=auto runs the checked-in llama-chat.rzb, which is older than
target/release/rayzor. Use the source path for both arms.

  cd /Users/amaterasu/Vibranium/rayzor/nue/examples/llama-chat
  for M in Qwen2.5-0.5B-Instruct-Q4_K_M qwen2.5-0.5b-instruct-q5_0; do
    ../../bench/plan/pins.sh env NUE_DUMP_Q4_GATES=1 NUE_PROFILE_POOL=1 \
      NUE_LLAMA_SILENT_STREAM=1 NUE_LLAMA_DUMP_OUTPUT=1 \
      ../../../target/release/rayzor run Main.hx --release --llvm -- \
      ~/models/qwen/$M.gguf "Explain what a B-tree is and why databases use one." \
      128 4096 0 1.0 > /tmp/before.$M.txt 2>&1
  done

Two models, deliberately: Q4_K_M is pure k-quant so canFuseRowwise is structurally
false (Q4Matmul.hx:1218-1221 accepts only INT8/Q8_0) and the SPLIT route is exercised;
q5_0 decodes to INT8 (GGUFLoader.hx case 6) so the FUSED route is exercised. One model
alone cannot distinguish "plan preserved" from "plan destroyed".

Do NOT set RAYZOR_DISABLE_TRACE — `[output]` is printed via trace() (Main.hx:281).
NUE_LLAMA_DUMP_OUTPUT is MANDATORY alongside SILENT_STREAM or generate() returns only
the prompt (GenerationLoop.hx:423-429, :380) and the diff passes on two empty strings.

--- 3. Health gate — run on EVERY captured artifact, before any number is believed ---
  grep -l 'LLVM upgrade failed\|E0100\|not registered\|SIGSEGV\|Segmentation' /tmp/before.*.txt && echo REFUSE
`[tier] LLVM upgrade failed` goes to STDERR (src/main.rs:1315-1319); every shipped gate
script discards it with 2>/dev/null, which is how a silent Cranelift fallback becomes
indistinguishable from a good run. Also assert the run exited 0 and printed `[done]` —
the two closest open bugs (exit 133 static->static trap; exit 139 new-module SIGSEGV)
emit NO diagnostic text at all, so a grep alone cannot see them.

--- 4. Extract the comparable block ---
Write /Users/amaterasu/Vibranium/rayzor/nue/bench/plan/extract.sh:

  #!/usr/bin/env bash
  f="$1"
  { grep -a '^\[nue-plan\]\|^\[q4-census\]\|^\[kv-cache\]\|^\[lm_head\]' "$f"
    grep -ao 'dispatches=[0-9]*' "$f"
    grep -ac '^\[q4-gate\]' "$f" | sed 's/^/q4gate_lines=/'
    sed -n '/^\[output\] /,$p' "$f"        # sed, NOT grep: [output] is the LAST
  } | sed -E 's/(band_ms|quant_ms)=[0-9.]+//g; s/w[0-9]+=[0-9.]+ms\/[0-9]+c//g'

DIFF MASK (measured, not assumed — two identical runs moved only these): strip
`band_ms=`, `quant_ms=`, and per-worker `wN=<ms>/<claims>`. Per-worker claim counts
are chunk-stealing artefacts (SpinPool.hx:170-184); only their SUM is deterministic.
The `[q4-gate]` line is COUNTED, not compared, because its print is a first-read side
effect (Q4Matmul.hx:113-121) and step 2 relocates it by design.

--- 5. Self-check the instrument (do this BEFORE trusting it) ---
Run arm A twice at HEAD with no code change and confirm extract.sh output is byte-
identical. Expected on Q4_K_M (survey-measured): `[nue-plan] fusion: triples fused=2568
split=0 | pairs fused=2568 split=0 | banding one-dispatch=5136 per-weight=0`,
`skipped-dispatcher triple=0 pair=0`, `dispatches=12971`, and exactly ONE `[q4-gate]` line.
If the two runs differ, STOP — the harness is not an oracle yet.

--- 6. ARM B — the prefill oracle (bench/eval, run DIRECTLY, not gate.sh --ab) ---
gate.sh --ab exports one env var into the SAME binary (gate.sh:78) and discards stderr
(:67), so it can never compare two code trees. Drive Main.hx directly:

  cd /Users/amaterasu/Vibranium/rayzor/nue/bench/eval
  for M in Qwen2.5-0.5B-Instruct-Q4_K_M qwen2.5-0.5b-instruct-q5_0; do
    ../plan/pins.sh ../../../target/release/rayzor run Main.hx --release --llvm \
      --safety-warnings off -- ~/models/qwen/$M.gguf corpus.txt \
      --chunk 256 --max-chunks 6 --dump /tmp/before.eval.$M.tsv > /tmp/before.eval.$M.txt 2>&1
  done

Then after each code step, re-run into /tmp/after.eval.$M.tsv and join:

  awk -F'\t' 'NR==FNR{a[$1]=$2;next} ($1 in a){n++; if(a[$1]==$2) same++}
              END{printf "top1=%.2f%% over %d\n", 100*same/n, n}' \
      /tmp/before.eval.$M.tsv /tmp/after.eval.$M.tsv

REQUIRED: top1 = 100.00% on BOTH models. This is teacher-forced from an empty cache
(bench/eval/gate.sh:16-18) so it exercises PREFILL, which the temp-0 text diff barely
touches; it removes the sampler, the RNG, the rep-penalty window and the 8-gram table;
and it is deterministic under contention. The argmax column is an exact Int; the NLL
column is rounded to 6dp (bench/eval/Main.hx:30-33) so it is a numeric check only.
Note bench/eval refuses when prefill is reduced to the last row (Main.hx:131-137) —
that is why NUE_PREFILL_LAST_LOGITS=0 is pinned.

--- 7. What is NOT baselined ---
No throughput number. This box currently fails its own pre-flight (memory pressure 2,
42% free against bench/mistral/gate.sh:68-75's MIN_FREE_PCT=45). Stage 1 makes no
timing claim; correctness and counter arms are contention-immune and were verified so
(NUE_MATMUL_WORKERS=8 vs =2 gave identical text and identical dispatches=11882).

## Steps

### Step 1 — landable commit

**Do:** Create the harness and capture the BEFORE baseline. Add `nue/bench/plan/pins.sh` (alias scrub + full pin list) and `nue/bench/plan/extract.sh` (diff mask). Run arm A and arm B on BOTH models at HEAD. Then run arm A a SECOND time at HEAD with zero code change and prove extract.sh output is byte-identical — if it is not, the instrument is not an oracle and no later step can be judged.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/bench/plan/pins.sh`
- `/Users/amaterasu/Vibranium/rayzor/nue/bench/plan/extract.sh`

**Verify:** Two HEAD runs of arm A produce byte-identical extract.sh output on both Qwen2.5-0.5B-Instruct-Q4_K_M and qwen2.5-0.5b-instruct-q5_0. Expected on Q4_K_M: `triples fused=2568 split=0`, `skipped-dispatcher triple=0 pair=0`, `dispatches=12971`, exactly ONE `[q4-gate]` line. Health gate clean on every artifact (no `LLVM upgrade failed`/`E0100`/`not registered`/`SIGSEGV`, exit 0, `[done]` present). /tmp/before.eval.*.tsv exist and are non-empty.

### Step 2 — landable commit

**Do:** Add `nue/nue/arch/NuePlan.hx` — ONE class, `package nue.arch` (co-located with its builder LlamaArch.hx; NOT a new package, NOT a new directory). Contents: (a) a policy block of PRIMITIVE public fields; (b) parallel `Array<Int>` node arrays `op/phase/layer/a/b/c` plus `Array<String> label`; (c) `public function new()`, `public function addNode(op:Int, phase:Int, layer:Int, a:Int, b:Int, c:Int, label:String):Void`, `public function dump():Void`. Opcodes are PRIVATE `static inline var` compared only inside this file (nue has ZERO cross-module reads of a public static inline var). ZERO mutable statics. Dump prefix is `[nue-graph]` — NOT `[nue-plan]`, which Q4Matmul.hx:352/358/364/373 already owns. Prefer instance methods over static→static helpers (open bug: same-class static→static in an import-compiled module traps at exit 133 with no diagnostic). In `LlamaArch.build()`, after `model.spinPool = sp;` (line 189) and before `return model;`, read the EXISTING flag with Q4Matmul.hx:186-191's exact truthiness and, only when truthy, construct a LOCAL NuePlan, fill it, `dump()` it, and drop it. NO field is added to LlamaModel. Record ONLY facts that need no new gate call: layer count and node kinds; per-projection weight scheme ordinal (from the QTensor already in scope in buildBlock); the BUILT cache kind read from `cache.useQ8H`/`cache.useQ8` (KVCache.hx:63-82 downgrades on `headDim%32!=0` and again on plugin alloc failure, and GQAttention.hx:260/295 branches on the built value — LlamaArch already reads these at :207); post-requant lm_head scheme (LlamaArch.hx:170-179 can turn a tied Q6_K head into Q4_K_M); rope style from `rope.neox`; head counts; and the policy fields sourced from accessors LlamaArch ALREADY calls (Linear.useHaxeMatmul at :94, the five pool accessors at :131/:134) plus the four env expressions moved character-for-character from :87-88, :95-105, :171, :269. Deliberately does NOT call useFusedMatmul/canFuseRowwise, so the `[q4-gate]` line provably does not move. Because there is no LlamaModel field, `prefillHandle` is unknown here — dump `graph_prefill=deferred`.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/NuePlan.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaArch.hx`

**Verify:** Arm A on both models: extract.sh output identical to BEFORE except for added `[nue-graph]` lines; `q4gate_lines=1` unchanged; `dispatches=` unchanged; `[output]` byte-identical. Arm B top1=100.00% on both models. Health gate clean. Assert the dump lists exactly `meta.numLayers` decoder layers and that the cache kind printed matches the `[kv-cache]` line LlamaArch already emits at :206-214. Run with NUE_DUMP_Q4_GATES UNSET and confirm zero `[nue-graph]` output (the shipping path constructs nothing).

### Step 3 — landable commit

**Do:** Add the four lowered fields and the VERIFY scaffold — planner computes, live expression still EXECUTES. Append (never insert; KVCache.hx:42-48: importers resolve fields by declaration order) `public var planHaxeMat:Int = 0;` and `public var planFusedQkv:Int = 0;` to GQAttention after `scale:Float` (line 51, the last instance field, ahead of the statics at :62-65), and `public var planHaxeMat:Int = 0;` / `public var planFusedPair:Int = 0;` to SwiGLU after `down:Linear` (line 25). State encoding, matching the tree's own load-bearing idiom (Linear.hx:65-70, FlashDecode.hx:23-25): 0=unplanned, 1=on, 2=off, 3=on+verify, 4=off+verify. Zero is unplanned, so the four instances built outside the planner — examples/tiny-transformer/Main.hx:148,152 and examples/transformer-pieces/Main.hx:34,112 — keep today's behaviour with NO edit. In `LlamaArch.buildBlock`: bind SwiGLU's three Linears to locals BEFORE `new SwiGLU(...)` at :234-238 (the route depends on `gate.qweight`/`up.qweight`, and per-layer promotion means Q6_K can appear on some layers, so the route is PER LAYER not per model). Then compute both routes with the expressions copied VERBATIM, preserving short-circuit order — for QKV `!haxeMat || Q4Matmul.useFusedMatmul() || Q4Matmul.canFuseRowwise(qProj.qweight, kProj.qweight, vProj.qweight)` (GQAttention.hx:161-162), for the FFN pair `Q4Matmul.useFusedMatmul() || Q4Matmul.canFuseRowwise(gwq, uwq, null)` (SwiGLU.hx:56-57) — and assign `attn.planHaxeMat/planFusedQkv` and `ffn.planHaxeMat/planFusedPair` post-construction (the proven-safe shape already used at LlamaArch.hx:72 `rope.neox = ...` and :180 `lmHead.pool = sp`). The two sites MUST keep separate expressions: GQAttention's `!useHaxeMat ||` makes fusedMat unconditionally TRUE under NUE_MATMUL=0 and routes to the FFI `fusedQkvIntoArr` (:166-172), while SwiGLU's `useHaxeMat &&` short-circuits to the two-Linear split — collapsing them silently changes the NUE_MATMUL=0 arm. Add ONE new gate, read once in LlamaArch.build and never on the hot path: `NUE_PLAN_VERIFY` / `RAYZOR_PLAN_VERIFY`, truthy, default OFF; when on the planner writes states 3/4 instead of 1/2. In GQAttention.forward and SwiGLU.forward, when the field is >= 3, evaluate the live expression as today, compare, `Sys.println` `[nue-graph] MISMATCH <site> layer=<blockName-free id>` on disagreement, and USE THE LIVE VALUE. Leave `noteFusionSite` (GQAttention.hx:163, SwiGLU.hx:55) firing every forward with byte-identical arguments.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/GQAttention.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/SwiGLU.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaArch.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/NuePlan.hx`

**Verify:** THE SHADOW MATRIX — the step that converts 'it matched once' into proof. With NUE_PLAN_VERIFY=1, sweep the full cross-product NUE_MATMUL x NUE_FUSED_MATMUL x NUE_FUSED_ROWWISE x NUE_FUSED_DISPATCH x NUE_HAXE_INT8 x NUE_HAXE_Q8_0 (64 runs) on BOTH models with a short generation, and require ZERO `[nue-graph] MISMATCH` lines across all 128 runs. Then arm A with NUE_PLAN_VERIFY unset: extract.sh identical to step 2 EXCEPT the `[q4-gate]` line, whose position moves (its print is a first-read side effect at Q4Matmul.hx:113-121 and the first read relocates from the first attention forward during warm() to build time) — record `q4gate_lines` before/after and require the COUNT to stay 1; a count of 2 means the static duplicated for the new reader and must be investigated, not waved through. Arm B top1=100.00% on both models. Health gate clean.

### Step 4 — landable commit

**Do:** Flip GQAttention to CONSUME the plan. Planner writes states 1/2 for `planHaxeMat`/`planFusedQkv` (verify mode still available). In GQAttention.forward, when `planHaxeMat != 0`, replace `var useHaxeMat = Linear.useHaxeMatmul();` (line 156), the `useFusedMat` expression (:161-162), and the SECOND redundant read `if (Linear.useHaxeMatmul())` (:166) with the decoded field values. All three consumers must read the SAME bit. When `planHaxeMat == 0` the original expressions run byte-unchanged. Keep `noteFusionSite(true, qWq != null && kWq != null && vWq != null, useHaxeMat)` at :163 firing every forward with the same three argument VALUES — it feeds `_planSiteTriple` and hence `sites - fused - split` (Q4Matmul.hx:364-372), the only counter that detects a site re-routed so it never reaches the dispatcher. Do NOT touch the five-tier ladder from :260 down, the null-fallthroughs (:260-345, including the nested `flashAttnDecodeQ8Host` -> `flashAttnDecodeQ8` fallback at :301-307), or any clone/free. Do NOT hoist `FlashDecode.enabled()`/`batchMax()` in this step.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/GQAttention.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaArch.hx`

**Verify:** Arm A on both models: extract.sh byte-identical to step 3's AFTER, including `[output]`, every `[q4-census]` line, the `sites:` line with `haxe-matmul-off-at-site=0` and `skipped-dispatcher triple=0 pair=0`, and `dispatches=`. Re-run the 128-run shadow matrix with NUE_PLAN_VERIFY=1 (now comparing a consumed value against the live one) — zero MISMATCH. Arm B top1=100.00% on both models. Health gate clean. Also confirm examples/transformer-pieces and examples/tiny-transformer still run to completion (they leave the field at 0).

### Step 5 — landable commit

**Do:** Flip SwiGLU to CONSUME the plan — separately, so a divergence is attributable to attention or to FFN, never to both. Replace `var useHaxeMat = Linear.useHaxeMatmul();` (SwiGLU.hx:54) and the fusion condition (:56-57) with the decoded fields when `planHaxeMat != 0`. Keep `noteFusionSite(false, gwq != null && uwq != null, useHaxeMat)` at :55 unchanged. Leave the siluMul/down/free sequence at :67-72 untouched.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/SwiGLU.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaArch.hx`

**Verify:** Same as the previous step: arm A byte-identical, 128-run shadow matrix clean, arm B top1=100.00% on both models, health gate clean, both standalone examples still run.

### Step 6 — landable commit

**Do:** Hoist the two per-block-per-token debug gates. Append `public var planDbgShape:Int = 0;` and `public var planDecodeSplit:Int = 0;` to LlamaBlock after `blockName:String` (line 18, ahead of `static var _debugShapes` at :19). Planner writes them from `LlamaBlock.debugShapeMode()`'s resolved value and `Q4Matmul.decodeSplitEnabled()`. In LlamaBlock.forward, when the field is non-zero use it instead of calling `debugShapeMode()` (:71) and the CROSS-MODULE `nue.Q4Matmul.decodeSplitEnabled()` (:81). The 0-sentinel cannot collide: `debugShapeMode()` returns only 1, 2 or 3 (LlamaBlock.hx:37-47) and never 0. Both hoisted values are process-constant, so this cannot change numerics or dispatch counts at all — which is exactly why it is the cheapest confirmation that the planner-writes-Int mechanism is sound, and it removes one cross-module static call per block per token (24+ per decoded token on a 7B). The comment already at LlamaBlock.hx:77-80 records that a previous version of this instrumentation cost decode time in production runs.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/LlamaBlock.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaArch.hx`

**Verify:** Arm A byte-identical on both models (this step provably cannot move a counter). Additionally run with NUE_DUMP_BLOCK_SHAPES=trace and =1 and NUE_PROFILE_DECODE_SPLIT=1 and confirm the emitted trace/profile lines match a HEAD run under the same flags. Arm B top1=100.00%. Health gate clean.

### Step 7 — landable commit

**Do:** Remove the VERIFY scaffold. Delete the `NUE_PLAN_VERIFY` read from LlamaArch, delete the 3/4 states and the comparison branches from GQAttention.forward and SwiGLU.forward. Planner writes 1/2 only; 0 still means unplanned. This deletes the one gate this work introduced, so the shipped gate surface is exactly the pre-existing set.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/GQAttention.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/transformer/SwiGLU.hx`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaArch.hx`

**Verify:** Arm A byte-identical to the previous step on both models. Arm B top1=100.00%. `grep -rn NUE_PLAN_VERIFY nue/` returns nothing. Full-chain check: diff step 7's arm A output against the step-0 BEFORE capture — the ONLY permitted deltas across the entire Stage 1 are the added `[nue-graph]` block and the POSITION of the single `[q4-gate]` line. Health gate clean.

### Step 8 — landable commit

**Do:** Write down what landed and, more importantly, the constraints that shaped it — otherwise the next session re-litigates them. In ROADMAP.md under Stage 1: NuePlan lives in `nue.arch` (co-located with its builder) and is a build-time LOCAL, never a field, never a static, never read by Q4Matmul/Linear/GQAttention/SwiGLU; policy travels as tri-state Ints on the receiver the kernel already has; `Q4Matmul.matmul`/`matmulFused`/`noteFusionSite`/`dumpPlan` signatures are FROZEN; the seven private kernel gates stay kernel-owned and `Q4Matmul.dumpPlan()`'s gates line remains the sole authority for them; new instance fields APPEND ONLY (KVCache.hx:42-48); the 0-means-unplanned sentinel is load-bearing. In PERFORMANCE.md next to the existing benchmarking rules (:361-402): the full pin list, the two-arm protocol, and the measured diff mask (strip band_ms/quant_ms/wN; count `[q4-gate]` rather than comparing it; use `sed -n '/^\[output\] /,$p'` never grep; SILENT_STREAM requires DUMP_OUTPUT; NUE_FLASH_BATCH is a CAP so `=1` disables batched prefill; NUE_DUMP_TOPK is not an oracle; every arm must be grepped for `LLVM upgrade failed`). In ARCHITECTURE.md, replace the future-tense 'NueGraph execution plans' section (:221-232) with the shipped shape. Also fix the stale LlamaModel.hx:26 doc that calls the blocks TransformerBlocks when :50 types them `Array<LlamaBlock>`.

**Files:**
- `/Users/amaterasu/Vibranium/rayzor/nue/ROADMAP.md`
- `/Users/amaterasu/Vibranium/rayzor/nue/PERFORMANCE.md`
- `/Users/amaterasu/Vibranium/rayzor/nue/ARCHITECTURE.md`
- `/Users/amaterasu/Vibranium/rayzor/nue/nue/arch/LlamaModel.hx`

**Verify:** Docs-only apart from one comment fix. Re-run arm A once on Q4_K_M and confirm byte-identical output to step 7.

## Must not do

- Do NOT add a `Null<nue-defined-class>` instance field anywhere. Verified: all seven nullable instance fields in nue are rayzor-stdlib types (Linear.hx:39,40,45,50; Embedding.hx:34; LayerNorm.hx:20; LlamaModel.hx:37). A nue-owned class in that position would be the first, against an open corruption record whose failure mode is `field != null` reading TRUE while the pointer is garbage — so a defensive null guard does not save it.
- Do NOT add `LlamaModel.plan` or any field holding the plan object. The plan is a LOCAL in LlamaArch.build(). Accept `graph_prefill=deferred` in the dump (prefillHandle is set later by GGUFLoader.hx:88-105).
- Do NOT edit /Users/amaterasu/Vibranium/rayzor/nue/nue/Q4Matmul.hx at all — not the signatures, not the arity, not the parameter types of `matmul` (:1236), `matmulFused` (:2058), `noteFusionSite` (:244) or `dumpPlan` (:339). Three in-code records describe a declaration change on this class breaking an UNRELATED cross-module access, twice specifically `model.spinPool` (:332-338, :1412-1415, :1462-1465). Adding a `plan:` parameter to matmulFused is the single riskiest available edit and its failure mode is silent.
- Do NOT make `amxPrefill` (Q4Matmul.hx:78), `haxeAmx` (:97), `useHaxeInt8` (:161), `useHaxeQ8_0` (:149), `nrowBlock` (:39), `FlashDecode.shiftedQ` (:44) or `FlashDecode.usePool` (:60) public. Verified private. Making them public is a declaration change on the two classes carrying the E0100 notes and must be its own separately-verified commit if ever wanted.
- Do NOT create a static policy singleton in any module, and do NOT have Q4Matmul, Linear, GQAttention or SwiGLU read a static from a new module. Two independent in-code records say a cross-module OBJECT static reads garbage (Linear.hx:42-45, Q4Matmul.hx:2040-2042) — that is the brief's landmine verbatim.
- Do NOT create a new package or a new directory (`nue/nue/graph/`). NuePlan.hx goes in the existing `nue/nue/arch/`, `package nue.arch`, co-located with its builder — the documented fix for cross-package Array/struct field drift.
- Do NOT use `-1` as the unplanned sentinel, and do NOT use a bit-mask encoding shared as a comment across three files. 0 = unplanned, tri/penta-state Int, matching the idiom stated in four files. A zeroed cross-module duplicate must fall back to live evaluation, never decode to a valid-looking route.
- Do NOT read a `public static inline var` across a module boundary. Verified: nue contains ZERO such reads (ChatTemplate.hx:19-22, GGUFReader.hx:35-36, BertEmbedder.hx:76-80 are all same-file only). Keep every opcode/state constant private to the file that compares it.
- Do NOT edit examples/tiny-transformer/Main.hx or examples/transformer-pieces/Main.hx. They construct GQAttention/SwiGLU/LlamaModel directly (Main.hx:148,152,66 and :34,112) and must keep working unchanged via the 0-sentinel. If a step makes them need an edit, the encoding is wrong.
- Do NOT collapse GQAttention's and SwiGLU's fusion predicates into one shared expression. GQAttention.hx:161 `!useHaxeMat ||` makes fusedMat unconditionally TRUE under NUE_MATMUL=0 and routes to the FFI `fusedQkvIntoArr` (:166-172); SwiGLU.hx:56 `useHaxeMat &&` short-circuits to the split. Unifying them silently changes the NUE_MATMUL=0 arm.
- Do NOT move, remove, or change the arguments of `noteFusionSite` at GQAttention.hx:163 or SwiGLU.hx:55. It is the acceptance bar's own instrument and it feeds the `skipped-dispatcher` remainder (Q4Matmul.hx:364-372), the only counter that detects a site re-routed so it never reaches the dispatcher.
- Do NOT pin a fusion route, elide a tier, or touch the five-tier attention ladder (GQAttention.hx:260-345) or any null-fallthrough, including the nested `flashAttnDecodeQ8Host` -> `flashAttnDecodeQ8` fallback at :301-307. Three tiers decline at runtime and `baseLen + seqQ > rowStride` (FlashDecode.hx:152) is not plan-time knowable.
- Do NOT replace the `for (block in blocks)` loops at LlamaModel.hx:84-86 and :104-106 with a plan-driven walk, and do NOT route resetCache/cacheLen/cacheCapacity/rewindCache through the plan. Cut from Stage 1: one layer kind exists, so it buys nothing, and it puts a new indirection on an aliased in-place residual (LlamaBlock.hx:99 `x.addInto(attnOut)`) in a compiler with a recorded loop-drop miscompile.
- Do NOT normalise the gate polarities or truthiness conventions. Three coexist in LlamaArch.hx alone: `!(v=="0"||v=="false")` at :88, `!= "0"` at :171 (does NOT accept "false"), `== "1"` at :269. Transcribe each verbatim or call the owning accessor; never route two gates through one shared helper.
- Do NOT re-implement any gate the planner needs — call the owning accessor. The in-repo cost of duplication is BertEmbedder.hx:146 `AMX_MIN = 16` against Q4Matmul.hx:71's default of 128: same gate, drifted 8x. Treat any re-implementation as a review defect even when it currently agrees.
- Do NOT use `[nue-plan]` as the new dump prefix — Q4Matmul.hx:352/358/364/373 owns it and the acceptance diff greps it. Use `[nue-graph]`.
- Do NOT use `bench/eval/gate.sh --ab` as the prefill oracle. Verified: gate.sh:78 exports one env var into the SAME binary and :67 discards stderr — it cannot compare two code trees and would pass vacuously.
- Do NOT use `NUE_DUMP_TOPK` as an oracle (measured: 1 line in four of five identical runs, 8 in the fifth), do NOT set `NUE_LLAMA_SILENT_STREAM` without `NUE_LLAMA_DUMP_OUTPUT` (GenerationLoop.hx:423-429, :380 — the diff then compares two empty replies and PASSES), do NOT set `RAYZOR_DISABLE_TRACE` (`[output]` is printed via trace at Main.hx:281), do NOT write `NUE_FLASH_BATCH=1` (any number is a CAP, FlashDecode.hx:109-117), and do NOT extract `[output]` with grep (it is the last thing printed; use `sed -n '/^\[output\] /,$p'`).
- Do NOT run the baseline through `bench.sh` with its default `BUNDLE=auto` — it executes the checked-in llama-chat.rzb, which is older than target/release/rayzor.
- Do NOT trust a run without capturing stderr (`2>&1`) and grepping for `LLVM upgrade failed` (src/main.rs:1315-1319). And do NOT treat that grep as sufficient: also assert exit code 0 and that `[done]` printed, because the two nearest open bugs produce exit 133 and exit 139 with NO diagnostic text.
- Do NOT make any throughput claim from this work. Stage 1 is behaviour-identical by construction; the box currently fails bench/common.sh's own pre-flight (memory pressure 2, 42% free vs MIN_FREE_PCT=45).
- Do NOT batch these steps into one commit. The recorded E0100/SIGSEGV failures are non-local and cannot be bisected by inspection, so each step must be verified before the next begins.

## Open risks

- ID-SPACE PERTURBATION is the real hazard and it cannot be ruled out by reading. All three Q4Matmul E0100 records describe a DECLARATION change breaking an UNRELATED access elsewhere. This plan adds one class and six instance fields — six independent perturbations. Mitigated by staging (step 2 is a pure addition with no semantic change), by co-locating in an existing package, by appending fields, and by never touching Q4Matmul — but it is proven only by running. Recorded probes if something breaks: RAYZOR_ALLOC_DEBUG=1 (alloc-size/TypeId punning), RAYZOR_CTOR_DEBUG / RAYZOR_FIELD_DEBUG (ctor and field-slot binding).
- THE VERIFICATION RUNS ON macOS AND THE HAZARD IS HOST-DEPENDENT. The new-module SIGSEGV record is x86-only ('macOS runs fine'), and the same source produced different SymbolId numbering on Mac vs NUC. A green macOS run does NOT exonerate the new class; it only proves Mac's numbering got lucky. Steps 2-7 should each get an x86 build+run before the chain is called done, and until that happens the Stage 1 claim is 'clean on arm64 Mac', not 'clean'.
- THE `[q4-gate]` LINE COUNT IS THE ONE THING THE HARNESS CANNOT PRE-VALIDATE. Step 3 makes LlamaArch (nue.arch) a new reader of `Q4Matmul.useFusedMatmul()`. Exactly ONE line prints at HEAD, which implies nue + nue.transformer share one static storage. If nue.arch gets its own duplicate, a SECOND identical line appears and 'sort before diffing' does not absorb a line-count change. The harness counts the lines and a count of 2 must be investigated, not masked. Partial mitigation: routing through `canFuseRowwise` executes useFusedMatmul inside Q4Matmul's own compiled body (:1204), which may keep it on one storage — worth trying first if the count moves.
- THE PLAN COUNTERS MAY NOT BE COHERENT ACROSS MODULES, AND THIS WAS NEVER TESTED. `noteFusionSite` is written from nue.transformer but `_planSiteTriple` is read by dumpPlan in nue.Q4Matmul, while Q4Matmul.hx:143-145 and FlashDecode.hx:23-25 both state statics get cross-module DUPLICATED. If it has two storages, the `sites - fused - split` remainder (Q4Matmul.hx:370-372) can print wrong or negative. Force a known site count and check the printed number BEFORE leaning on `skipped-dispatcher` as the acceptance instrument. The new `[nue-graph]` dump reads instance fields and is immune, which is why it exists.
- 'IDENTICAL DISPATCH COUNTS' IS COARSER THAN THE BAR'S WORDING. There is exactly ONE counter — SpinPool cell 7 (SpinPool.hx:107, :373) — shared by all eleven parallelRows sites, it does NOT increment when the band runs inline (SpinPool.hx:306-310), and several sites bypass parallelRows below hardcoded thresholds (Q4Matmul.hx:516, :547, :747; FlashDecode.hx:310-313). It is cumulative with no reset API and includes the ~82 warm() forwards. Stage 1 is safe because it changes no kernel call, but this metric cannot distinguish a dispatch MOVING between node types from a genuine no-op, and no per-phase claim is possible today. Do not lean on it for any later policy change without adding tagged attribution.
- THE CENSUS IS NOT A DISPATCH COUNT AND HAS BLIND SPOTS: the rowwise fused path records THREE matmuls for ONE dispatch (Q4Matmul.hx:2103-2109 vs the single parallelRows at :1791); the k-quant fused paths call planFused but never census (:2157, :2221); and the fused AMX prefill branch (:2137-2142) counts neither a kernel nor a platform escape. Any harness equating `[q4-census] haxe=N` with dispatch count produces a false pass or fail.
- SIX GATES REMAIN KERNEL-OWNED, so the brief's 'ONE policy object' is delivered for every PLAN-VISIBLE gate but is not total. In particular `shiftedQ()` is read INSIDE the per-token kernel (FlashDecode.hx:254) and switches both the query quantiser (:286-293) and the band function (:300-308) — a genuine numerics selector the plan does not own — and turning it on additionally makes decodeBatch decline outright (:144), silently disabling batched prefill. That cross-gate interaction is documented in neither gate.
- `amxPrefill()` folds `Sys.systemName() == "Mac"` into its default (Q4Matmul.hx:78-86). If a later stage migrates it into the policy object, the platform term must migrate with it — a policy resolved on this Mac and reused on x86 would route Q4_K_M prefill into a path the band-kernel choice does not cover.
- THE PREFILL ORACLE HAS A HOLE: bench/eval refuses when prefill is reduced to the last row (bench/eval/Main.hx:131-137), so the CoreML graph-prefill path is unreachable by top-1 agreement. That path also BYPASSES the entire block stack and seeds the KV caches directly (LlamaModel.hx:181-217, :99-102), which is exactly why NUE_PREFILL=off is pinned — Stage 1 makes no claim about it.
- NOTHING HERE HAS BEEN BUILT OR RUN. Every file:line in this plan was read at HEAD and every disputed judge claim was checked against the source (direct constructions in the two examples, the absence of any Null<nue-class> field, the private/public split on all thirteen gate accessors, LlamaModel's last field, the `[nue-plan]` prefix collision, gate.sh --ab's single-binary semantics), but no compile and no execution was performed in this session.
- STAGE 2 FITNESS IS NOT ADDRESSED, DELIBERATELY. The lowered field is per-class, so a heterogeneous family (SSM/gated-attention interleave) needs its own fields and its own planner arm; `matmulFused` is hard-3 (Q4Matmul.hx:2058) and `noteFusionSite(isTriple:Bool, ...)` cannot COUNT a 4-way group; and `LlamaModel.cacheLen()` reads `blocks[0].attn.cache` (:237-242), which is 0 when layer 0 has no cache. Two cheap Stage-2 unlocks are available and were consciously deferred rather than smuggled in: widening `noteFusionSite`'s first parameter from Bool to Int (unchanged arity) and adding a group column to the plan IR. Both are declaration changes on the E0100-hazard class or on freshly-landed code, and neither belongs in a commit whose bar is bit-identity.
