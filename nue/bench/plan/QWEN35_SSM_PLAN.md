# Qwen3.5-9B gated-delta-net (Step 2 — the SSM block) — implementation plan

Produced by a survey/design/adversarial-judge pass over the real code, the real
GGUF, and two independent reference implementations. Every claim below carries a
file:line or a pasted command output. Every judge-found fatal flaw is folded in,
and each was **re-checked against the code in this session** rather than trusted.
Where evidence is thin it says so.

HEAD at time of writing: `b53c19c0`. The briefing describes step 1 as landing at
`a1157349`; that is four commits back. Re-verify against HEAD, not the briefing.

## Chosen design

**Port the already-proven fixture-driven kernel first, then grow outward to the
full-attention half, then the hybrid stack.** Ten landable commits. Steps 1–4 pin
the recurrence numerically against a two-implementation oracle before any model,
session, registry or GGUF plumbing exists. Steps 5–6 do the same for partial RoPE
and the fused-gate attention layer — the half of the model that *looks* familiar
and is therefore the half most likely to be ported wrong by pattern-matching
`GQAttention`. Steps 7–9 build the hybrid stack and fix the four silent
session-lifecycle degradations that layer-wise taps structurally cannot see.
Step 10 is perf, alone, moving no tap.

### Rationale

The novel math is already **demonstrated in Haxe, not proposed**. I re-ran it at
HEAD this session:

```
$ target/release/rayzor run Main.hx --llvm --release --no-cache -- <fixture> none
gdn decode probe   break=none
  ok   conv_raw     maxrel=7.322419608619601e-08   gate=0.0001
  ok   conv_out     maxrel=7.322419608619601e-08
  ok   q(l2+scale)  maxrel=1.2838617465775027e-07
  ok   k(l2)        maxrel=1.3858992923806097e-07
  ok   v            maxrel=7.322419608619601e-08
  ok   beta         maxrel=5.994329484194542e-08
  ok   gamma        maxrel=2.980236324395863e-08
  ok   core         maxrel=1.4901091028082893e-07
  ok   state_out    maxrel=1.3225274027677855e-07
  ok   gated        maxrel=2.3534398288390282e-07
PASS                                              (exit 0)
```

and the five structural controls, each of which **must** fail (exit codes checked
individually, not inferred):

| break | first failing tap | max rel | exit |
|---|---|---|---|
| `convtaps` (reverse the 4 conv taps) | `conv_raw` | 9.758e-01 | 1 |
| `interleave` (`hk = hv>>1` instead of `hv % 16`) | `q` | 1.0923e+00 | 1 |
| `alog` (`g = -exp(ssm_a)*softplus(...)`) | `gamma` | 7.135e-01 | 1 |
| `nodecay` (skip `S *= γ`) | `core` | 2.181e-01 | 1 |
| `normplus1` (read `ssm_norm` as `1+w`) | `gated` | 1.5494e+00 | 1 |

Source: `/private/tmp/claude-501/-Users-amaterasu-Vibranium-rayzor/0e5285ee-bd3c-4d54-a191-d5a96a4ca7c6/scratchpad/gdnprobe/Main.hx` (251 lines).
**Port that file; do not re-derive it from the spec.** Re-deriving re-runs every
risk it has already retired.

The Python oracle also reproduces at HEAD, in 2.0 s:

```
$ mlxenv/bin/python verify_goldens.py
blk  case                                         max rel diff vs mlx_lm
0    golden (HF l2norm eps) vs mlx_lm             3.686e-04
0    same, matching MLX eps convention            9.666e-07
0    CONTROL repeat_interleave (HF head order)    3.809e-01
0    CONTROL ssm_a read as A_log                  2.294e-02
0    CONTROL conv taps reversed                   3.109e-01
0    CONTROL ssm_norm as (1+w)                    1.578e+00
1    golden (HF l2norm eps) vs mlx_lm             6.240e-03
1    same, matching MLX eps convention            1.239e-06
1    CONTROL repeat_interleave (HF head order)    8.678e-01
1    CONTROL ssm_a read as A_log                  1.958e-01
1    CONTROL conv taps reversed                   1.026e+00
1    CONTROL ssm_norm as (1+w)                    1.186e+00
```

Two independent implementations agree at 9.7e-07 / 1.2e-06 on the **real
Qwen3.5-9B Q4_K_M weights**: a plain-numpy transcription of transformers 5.15.1
(`ref_gdn.py`) and `mlx_lm.models.qwen3_5.GatedDeltaNet` (Apple's own, already in
the venv). A single transcription can be confidently wrong; two cannot be wrong
the same way. **Do not install torch** — it is not needed, and `mx.load()` cannot
read k-quant GGUF anyway (`RuntimeError [load_gguf] gguf_tensor_to_f16 failed`),
so nue's own dequant stays the single source of weights.

### Judge-found fatal flaws, each re-checked against the code

1. **A proposed "new per-tensor decode entry point" rests on a FALSE premise —
   deleted.** The claim was that `GGUFLoader.populateTensor` and
   `tensorsFromReader` are `private static` so no test can reuse nue's dequant.
   Both *are* private (`nue/nue/loader/GGUFLoader.hx:325`, `:345`) — but
   `public function readNamedTensors(path:String):NamedTensorMap` already exists
   at **`GGUFLoader.hx:57-61`** and calls `tensorsFromReader` for you. Adding new
   public loader surface on top of a known silent trap (same-class static→static
   in an import-compiled module, exit 133) for a capability that already ships is
   pure risk. **Use `readNamedTensors`.**

2. **Partial RoPE is MANDATORY and is structurally impossible with existing ops.**
   `rope_impl` computes `let half = head_dim / 2;` (`rayzor-tensors/src/tensor.rs:3072`),
   gates on `if cos_shape[1] != half || sin_shape[1] != half { return 0 }` (`:3081`),
   and loops `for i in 0..half` (`:3107`). For `head_dim=256` that pairs
   `(i, i+128)` across all 256 lanes. Qwen3.5 rotates only lanes `[0,64)` pairing
   `(i, i+32)`. No cos/sin table can express that, and the failure is a returned
   **0 → null Tensor**, not a throw. Any plan that stops at the SSM kernel never
   reaches Qwen3.5 inference. Step 5 does it in pure Haxe, isolated, on one tensor.

3. **`out.y` did not exist in the decode fixture.** Audited
   `scratchpad/decfix/`: **22** files (not 21), 4,720,728 bytes, taps
   `in.{qkv,z,b,a,state,conv_state}`, `par.{conv1d,A,dt_bias,ssm_norm}`,
   `out.{conv_raw,conv_out,q,k,v,beta,g,core,state,gated,conv_state}` — **no
   `out.y`**, because it stops before `ssm_out`. The *prefill* oracle at
   `scratchpad/oracle/goldens/` does have `blk0.y`/`blk1.y`. Fix: one merged
   generator emits both cases, and the whole-layer step gates on `y`.

4. **The decode fixture manifest pinned nothing.** `decfix/manifest.json` contains
   exactly `["taps","cfg"]` — no model path, no hash, no token ids. Only
   `oracle/goldens/manifest.json` carries
   `["model","model_sha256_first1MB","tokens","cfg","layers","taps","checks","gen_seconds"]`
   (verified: sha256-of-first-1MB `ef041c00…0e32d`, tokens
   `[9707,11,4438,525,498,3351,30,358]`, checks
   `layer0_prefill_vs_split_maxrel=2.974e-07`, `layer1=1.444e-07`). The two must
   be merged before any hash-assert step is real.

5. **Gate PER TAP, never on block output or logits.** The weakest control is
   `alog` — the single most likely porting bug, because it is what HF's source
   literally says. At the *block-output* level it moves only **2.294e-02**
   (blk0), i.e. 2.3× above a 1e-2 gate. Per tap it moves `gamma` by **7.135e-01**.
   A logits- or `y`-level gate cannot separate it from quantisation noise and it
   would read as fluent English. Gate every tap at 1e-4; never loosen past ~1e-3.

6. **A stubbed-identity stack's "top-1 must differ from golden" is decorative.**
   With identity blocks the stack is `embed → outputNorm → lm_head`, so that
   check passes no matter how broken the plumbing is. Replaced with checks that
   actually flip: the layer-visit trace, the poison/reset control, and the
   allocation census. Also: this plan does not build an all-stub stack at all —
   by the time the stack is assembled (step 8) both block types are real and
   individually pinned.

7. **Do not register `qwen35` in `ArchRegistry.withDefaults()` until the model is
   real.** `GGUFLoader.registry` is `public var` (`GGUFLoader.hx:63`) and both
   build paths honour it — `var reg = (registry != null) ? registry : ArchRegistry.withDefaults();`
   at `:70` and `:184`. A test injects its own registry. Registering early ships a
   registered arch producing garbage for several commits, and registering is
   specifically the one-line mistake to avoid: `LlamaArch.validate()` **would
   pass** this model (`headDim*numHeads == hiddenSize`, `256*16 == 4096`,
   `LlamaArch.hx:51-62`) and then die at `LlamaArch.hx:287` on
   `prefix + "ffn_norm.weight"` — a tensor this file does not contain
   (`ffn_norm` count = **0**; the pre-FFN norm is `post_attention_norm.weight`,
   present on all 33 blocks).

8. **The four silent session-lifecycle degradations are real and none is testable
   by a layer-wise tap.** Verified in source:
   - `LlamaModel.resetCache()` (`:230-237`) skips any layer whose `attn`/`cache`
     is null → 24 SSM states would never clear between prompts.
   - `cacheLen()` (`:239-244`) and `cacheCapacity()` (`:246-254`) answer from
     `blocks[0]` only → both return 0 for a hybrid whose layer 0 is SSM.
   - `GenerationLoop.hx:221`: `if (_cacheCap > 0 && model.cacheLen() + 1 > _cacheCap)`
     — capacity 0 **disables the context stop entirely** rather than failing.
   - `rewindCache()` (`:256-263`) has the same null-skip, on a state with no
     truncatable prefix.
   Each produces fluent output. Step 8 attacks them with a poison/reset control
   that genuinely flips.

9. **Q5_K is re-encoded lossily and it lands exactly on the attention layers.**
   `GGUFLoader.hx:388-391` case 13 → `decodeQ5KToQ4KM`. Census of the file: all
   **16** Q5_K tensors are `attn_k.weight` and `attn_output.weight` on the 8 body
   full-attention layers; **zero** on SSM layers, and blk.32 is entirely Q8_0.
   This puts an irreducible floor under any end-to-end comparison that no correct
   code removes — but it also makes the diagnosis decisive (see step 9).

## Corrections to the briefing (each verified against the file)

Command: `mlxenv/bin/python` over `~/models/qwen/Qwen3.5-9B-Q4_K_M.gguf` using
`scratchpad/oracle/gguf.py`.

- **`attn_gate.weight` is on the 24 SSM layers, not the 8 full-attention ones.**
  `attn_gate` layers = `[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]`
  — byte-identical to the `ssm_a` layer list. It is HF's `in_proj_z` `[4096→4096]`,
  the SiLU gate consumed by the gated RMSNorm.
- **The full-attention output gate is FUSED into `attn_q.weight`,** which is
  `(8192, 4096)` = 16 heads × (256 query + 256 gate), **interleaved per head**.
  `mlx_lm/models/qwen3_next.py:129-133` does
  `mx.split(q_proj_output.reshape(B, L, num_heads, -1), 2, axis=-1)` — the split
  happens *after* the per-head reshape, so a flat `[0:4096]|[4096:8192]` split is
  a silent miscompile. Applied at `:158` as `o_proj(output * mx.sigmoid(gate))`.
- **Nine blocks carry full-attention tensors, not eight.** `attn_q` layers =
  `[3,7,11,15,19,23,27,31,32]`. Block 32 is a **complete** decoder layer
  (attn_q/k/v/output, q_norm, k_norm, attn_norm, post_attention_norm, full
  ffn_gate/up/down, all Q8_0) **plus** the four `nextn.*` tensors — **258.56 MB**,
  4.2% of the file. Skipping it by tensor-name presence would still build it.
  Skip **by index**: `bodyLayers = block_count - nextn_predict_layers = 32`.
- **There is no `ffn_norm.weight` anywhere.** `post_attention_norm.weight` count
  = 33; `ffn_norm` count = 0. `attn_norm.weight` is likewise on all 33 blocks.
- **The `ssm.*` key names are misleading.** `ssm.time_step_rank=32` is
  `linear_num_value_heads` (Hv) — this architecture has **no dt projection at
  all**. `ssm.group_count=16` is `linear_num_key_heads` (Hk).
  `ssm.state_size=128` is `linear_key_head_dim == linear_value_head_dim`.
  `ssm.inner_size=4096` is `value_dim = Hv·Dv`. Source:
  `scratchpad/oracle/qwen_conv.py:376-380`.
- **GGUF `ssm_a` is NOT `A_log`.** `qwen_conv.py:388-389`:
  `if name.endswith(".A_log"): data_torch = -torch.exp(data_torch)`. Census over
  all 24 SSM layers: `n=768, >=0 count=0, min=-76.9957, max=-0.008926`.
  Transcribing HF's line verbatim is a silent wrong answer.
- **The `+1` of the zero-centred RMSNorm is already folded into every norm
  weight EXCEPT `ssm_norm`.** `qwen_conv.py:394-395`:
  `elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"): data_torch = data_torch + 1`.
  Measured: `blk.0.attn_norm` mean 1.03306, `blk.3.attn_q_norm` mean 1.34101,
  `output_norm` mean 2.13984 — versus `blk.0.ssm_norm` mean **0.88117**, max
  0.9626 (entirely below 1). nue's plain `x̂·w` (`RMSNorm.hx:42`
  `x.rmsNormWeight(weight, eps)`) is correct for the first family and correct for
  `ssm_norm` too — just do not add 1 anywhere.
- **RoPE is PARTIAL and nue reads none of the metadata for it.**
  `qwen35.rope.dimension_count = 64` against `attention.key_length = 256`;
  `rope.freq_base = 1e7`; `rope.dimension_sections = [11 11 10 0]`;
  `nextn_predict_layers = 1`. `ARCH_EXTRA_KEYS` (`GGUFLoader.hx:237-246`) lists
  exactly 8 keys and contains neither `rope.dimension_count` nor
  `nextn_predict_layers`.
- **The V-head order in GGUF is TILED, not grouped.**
  `qwen_conv.py:455-470` `_reorder_v_heads`, applied to the qkv V-rows,
  `in_proj_z`, `in_proj_a/b`, `A_log`, `dt_bias`, conv1d V-channels and
  `out_proj` columns. So the k-head serving value head `hv` is `hv % Hk`.
  `rayzor_tensor_expand_kv_heads_axis1_f32` computes `let kvh = qh / repeats;`
  (`tensor.rs:1897`) — **repeat-interleave, the wrong permutation here.** Do not
  use `Tensor.expandKvHeadsAxis1` on the SSM path.
- Minor: tokenizer is `gpt2`/`qwen35` BPE (not SentencePiece), `context_length`
  is 262144, `output.weight` is present so embeddings are **untied**, and
  `general.tags = ['image-text-to-text']` while the GGUF ships zero vision
  tensors (442 total, all `blk.*`/`token_embd`/`output`/`output_norm`).

**Open, cannot be settled from the file:** `ssm_alpha ↔ in_proj_a` and
`ssm_beta ↔ in_proj_b` are both `(32, 4096)` F32, so a swap is shape-silent.
The alphabetical correspondence matches the converter's naming and the oracle
agrees numerically at 9.7e-07 with that assignment — which *is* the proof, but
only because the whole block matched. Keep the `alphabeta` control (step 4).

---

## THE MATH

All shapes for this checkpoint, batch 1, everything f32.

### Constants

```
D    = 4096   hidden_size
Hk   = 16     linear_num_key_heads        (GGUF ssm.group_count)
Hv   = 32     linear_num_value_heads      (GGUF ssm.time_step_rank)
Dk   = 128    linear_key_head_dim         (GGUF ssm.state_size)
Dv   = 128    linear_value_head_dim       (= ssm.inner_size / Hv)
rep  = Hv/Hk = 2
key_dim   = Hk*Dk = 2048
value_dim = Hv*Dv = 4096
C    = conv_dim = 2*key_dim + value_dim = 8192
K    = 4      ssm.conv_kernel
eps  = 9.999999974752427e-07   (qwen35.attention.layer_norm_rms_epsilon)
l2eps= 1e-6                    (HF hardcodes this; it is NOT the same as eps above)
```

### Per-layer weights, GGUF-oriented `[out, in]`

```
attn_norm.weight            [4096]         F32
attn_qkv.weight             [8192, 4096]   Q6_K   = in_proj_qkv
attn_gate.weight            [4096, 4096]   Q4_K   = in_proj_z
ssm_beta.weight             [32, 4096]     F32    = in_proj_b   (write strength)
ssm_alpha.weight            [32, 4096]     F32    = in_proj_a   (decay input)
ssm_conv1d.weight           [8192, 4]      F32    (row c = channel c's 4 taps, contiguous)
ssm_a                       [32]           F32    = -exp(A_log) = -A, all NEGATIVE
ssm_dt.bias                 [32]           F32
ssm_norm.weight             [128]          F32    (plain w, NO +1)
ssm_out.weight              [4096, 4096]   Q8_0 on 12 layers, Q4_K on the other 12
post_attention_norm.weight  [4096]         F32
ffn_gate/up.weight          [12288, 4096]  Q4_K
ffn_down.weight             [4096, 12288]  Q6_K on 16, Q4_K on 16
```

`ssm_conv1d.weight` GGUF `ne=[4,8192]`; `GGUFLoader.orientDims` yields `[8192,4]`,
so each channel's four taps are already contiguous — **no transpose needed.**

### One SSM layer, prefill form (S tokens)

```
x        = RMSNorm(h, attn_norm.weight, eps)              [S, 4096]

qkv      = x @ Wqkv^T                                     [S, 8192]
z        = x @ Wz^T                                       [S, 4096]
b        = x @ Wb^T                                       [S, 32]
a        = x @ Wa^T                                       [S, 32]

# depthwise CAUSAL conv over TIME, C=8192 independent channels, groups=C, no bias
# left context = conv_state [K-1=3, C]  (zeros for a fresh prompt)
buf      = concat(conv_state, qkv)                        [3+S, 8192]
conv_raw[t][c] = sum_{j=0..3}  Wconv[c][j] * buf[t+j][c]
                 # equivalently  sum_{j} Wconv[c][K-1-j] * qkv[t-j][c]
                 # TAP INDEX 3 MULTIPLIES THE CURRENT TOKEN
conv_state' = buf[-3:]                                    [3, 8192]
conv_out = SiLU(conv_raw)                                 [S, 8192]

q_pre    = conv_out[:,      0 : 2048 ].reshape(S, 16, 128)
k_pre    = conv_out[:,   2048 : 4096 ].reshape(S, 16, 128)
v        = conv_out[:,   4096 : 8192 ].reshape(S, 32, 128)

# head expand: TILE (GGUF), k-head for value head hv is  hv % 16
q_t[hv]  = q_pre[hv % 16]  ;  k_t[hv] = k_pre[hv % 16]    [S, 32, 128]

# l2norm is rsqrt(SUM(x^2) + 1e-6) — NOT rmsNorm's mean form
q[hv]    = q_t[hv] * rsqrt(sum(q_t[hv]^2) + 1e-6) * (1/sqrt(128))
k[hv]    = k_t[hv] * rsqrt(sum(k_t[hv]^2) + 1e-6)          # k is NOT scaled

beta[hv] = sigmoid(b[hv])                                  in (0,1)
g[hv]    = ssm_a[hv] * softplus(a[hv] + dt_bias[hv])       <= 0   (ssm_a is already -A)
gamma[hv]= exp(g[hv])                                      in (0,1]
```

Then the recurrence, run for `t = 0 .. S-1`, carrying `S_state[hv]` of shape
`[Dk=128][Dv=128]`. **The order is strict** — decay, read, write, read-out:

```
for t in 0..S-1:
  for hv in 0..31:
    (1) DECAY    S[hv][dk][dv] *= gamma[t][hv]                       # per-head SCALAR
    (2) READ     m[dv]  = sum_dk  S[hv][dk][dv] * k[t][hv][dk]
    (3) DELTA    d[dv]  = (v[t][hv][dv] - m[dv]) * beta[t][hv]
    (4) WRITE    S[hv][dk][dv] += k[t][hv][dk] * d[dv]               # rank-1 outer product
    (5) READOUT  core[t][hv][dv] = sum_dk  S[hv][dk][dv] * q[t][hv][dk]
```

Then gated RMSNorm per value head over `Dv`, and the output projection:

```
for each (t, hv):
  c = core[t][hv]                                          [128]
  c = c * rsqrt(mean(c^2) + eps)                           # normalise FIRST
  c = c * ssm_norm.weight                                  # PLAIN w, no +1
  c = c * SiLU(z[t] viewed as [32,128] at row hv)          # gate LAST
gated[t] = flatten(c over hv)                              [S, 4096]
y        = gated @ Wout^T                                  [S, 4096]

h  = h + y                                     (residual)
h  = h + SwiGLU(RMSNorm(h, post_attention_norm.weight, eps))
```

### Decode form (S = 1)

**Identical math**, different schedule and left-context source:

- conv: read the 3-deep per-channel ring `conv_state`, append `qkv_t`, emit one
  row, keep the last 3 columns. Formula unchanged.
- recurrence: run steps (1)–(5) exactly once, from the carried `S_state`.
- everything else is the same expression at `S = 1`.

HF additionally ships a chunked/WY-transform prefill (`chunk_size=64`). It is an
arithmetic-intensity optimisation, **not a different model** — HF's chunked and
recurrent paths agree at `2.701e-08` absolute / `6.622e-07` relative on the
output and `3.053e-07` on the state. `mlx_lm` ships only the sequential
recurrence for both paths. **Implement the recurrence only; chunk by loop order.**

### State sizes (constant in context)

```
recurrent  S_state  [Hv=32][Dk=128][Dv=128] f32 = 2,097,152 B = 2.00 MiB / layer
conv ring           [K-1=3][C=8192]        f32 =    98,304 B =   96 KiB / layer
per SSM layer                                   = 2,195,456 B = 2.094 MiB
x 24 SSM layers                                 = 52,690,944 B = 50.25 MiB
```

(HF keeps `conv_kernel_size = 4` columns; only `K-1 = 3` are load-bearing.
`cache_utils.py` `update_conv_state` keeps the last `conv_kernel_size` columns
and left-pads a fresh prefill.)

KV cache is needed for **only 8 layers**:
`2 * numKvHeads(4) * headDim(256) * 4 B = 8192 B/token/layer` F32 → 64 KiB/token,
256 MiB at ctx 4096. Q8 (`Q8Cache.hx:44-49`: `rowBytes = numKvHeads*(headDim>>5)*34`
= 1088, plus `scaleF32` and `sumI32` at `blocksPerRow*4` = 128 each) → 2688
B/token/layer → 21 KiB/token, 84 MiB at ctx 4096.

**Counter-intuitive and worth stating:** below **268 tokens** (F32) or **817
tokens** (Q8) of context, one SSM layer's state is *larger* than the KV it
replaces, and the 50.25 MiB floor is paid at session creation. The hybrid saves
memory only at long context, and N concurrent spokes multiply the floor.

### The full-attention layer (8 of them: 3, 7, 11, 15, 19, 23, 27, 31)

```
x   = RMSNorm(h, attn_norm.weight, eps)                    [S, 4096]
qg  = x @ Wq^T   -> view [S, 16, 512]                      # 16 heads x (256 q | 256 gate)
q   = qg[:, :, 0:256]                                      # PER HEAD, never a flat split
gate= qg[:, :, 256:512]  -> reshape [S, 4096]
k   = (x @ Wk^T).reshape(S, 4, 256)
v   = (x @ Wv^T).reshape(S, 4, 256)
q   = RMSNorm(q, attn_q_norm.weight, eps)                  # over 256, AFTER the split
k   = RMSNorm(k, attn_k_norm.weight, eps)
q,k = PartialRoPE(q, k, pos)                               # see below
ctx = softmax(q @ k^T * (256^-0.5)) @ v                    # scale uses the FULL 256
out = attn_output( ctx.reshape(S,4096) * sigmoid(gate) )
```

`scale = head_dim**-0.5 = 1/16` (`mlx_lm/models/qwen3_next.py:87`), applied over
the **full** 256, not the rotary 64. Gate applied **before** `attn_output`
(`qwen3_next.py:158`).

### PartialRoPE — settled numerically this session

Only lanes `[0, 64)` rotate, NeoX/half-split pairing `(i, i+32)` **inside those
64**; lanes `[64, 256)` pass through untouched; base `1e7`.
`mlx_lm` builds it as `initialize_rope(int(head_dim * partial_rotary_factor), base, traditional=False, ...)`
(`qwen3_next.py:113-119`), i.e. `mx.fast.rope(x, dims=64, traditional=False)`.

Measured against `mx.fast.rope` on a `[1,1,1,256]` vector at `pos=5`, `base=1e7`:

| candidate | max rel vs `mx.fast.rope` |
|---|---|
| **NeoX over 64 lanes, pairing (i, i+32)** | **1.151e-07** |
| interleaved (2i, 2i+1) over 64 lanes | 4.296e-01 |
| NeoX over the full 256 lanes | 9.371e-01 |
| lanes[64:256] vs input, max abs diff | 0.000e+00 |

**This also closes the open `bugs_rope_interleaved_for_gguf` question for this
model.** The qwen converter contains no q/k row permute — the only `permute`
calls are the qkvz transpose (`qwen_conv.py:412-423`) and `_reorder_v_heads`
(`:469`). NeoX/half-split applies directly to the GGUF weights. Keep the
`interleaved_rope` control anyway; it is now cheap and it is the check that
would catch a converter change.

The table builder is reusable as-is: `Tensor.ropeCosTable(64, maxSeqLen, 1e7)`
returns `[maxSeqLen, 32]` (`rayzor-tensors/haxe/rayzor/ds/Tensor.hx:462`,
`tensor.rs:3490-3491`) — exactly the 32 angles the 64-lane rotation needs. Only
the *application* has to be hand-written in Haxe.

---

## The oracle recipe

Everything below is torch-free and runs on what is already installed. Do **not**
install packages.

```
SP=/private/tmp/claude-501/-Users-amaterasu-Vibranium-rayzor/0e5285ee-bd3c-4d54-a191-d5a96a4ca7c6/scratchpad
PY=$SP/mlxenv/bin/python          # python 3.14, mlx 0.32.1, mlx_lm 0.31.3,
                                  # transformers 5.15.1, numpy 2.5.2
```

Note: `mlxenv` contains BOTH a `python3.13` and a `python3.14` site-packages
tree. `$SP/mlxenv/bin/python` is 3.14. Use that one path consistently or the
`mlx_lm` you read and the `mlx_lm` you import will differ.

Source files to move into the repo (all four already exist and run):

```
$SP/oracle/gguf.py             mmap GGUF reader + Q4_K/Q5_K/Q6_K/Q8_0/F16/F32 dequant
                               (block layouts mirrored from nue/nue/Q4Matmul.hx:432, :930)
$SP/oracle/ref_gdn.py          plain-numpy Qwen3_5GatedDeltaNet, layout='gguf'|'hf'
$SP/oracle/gen_ssm_goldens.py  emits the prefill fixture + the state-carry check
$SP/oracle/verify_goldens.py   numpy-vs-mlx_lm cross-check + 4 structural controls
```

Generate and verify:

```
cd /Users/amaterasu/Vibranium/rayzor/nue/tests/qwen35
$PY gen_goldens.py goldens     # ~0.84 s generate, 8.1 MB, 41 taps
$PY verify_goldens.py          # ~2.0 s, must exit 0
```

### Tolerances — calibrated, not guessed

| mode | what it compares | gate | why that number |
|---|---|---|---|
| **Oracle floor** | numpy vs `mlx_lm`, real weights | ≤ 1e-5 | measured 9.666e-07 (blk0), 1.239e-06 (blk1) |
| **Structural controls** | each deliberate break | ≥ 1e-2 | weakest measured is `alog` at 2.294e-02 |
| **Mode A** (inject the golden's own upstream tap; f32 path) | every Haxe tap | ≤ 1e-4 | 417× above the measured Haxe floor of 2.353e-07; 254× below the weakest control |
| **State carry** (`prefill(7)+decode(1)` vs `prefill(8)`) | state and y | ≤ 1e-5 | reference's own residual is 2.974e-07 / 1.444e-07 |
| **Mode B** (propagate through nue's *quantised* kernels) | per tap | cos ≥ 0.9999 **and** rms-rel ≤ 2e-2 | measured Q8_K activation-quant budget on `blk.0.attn_qkv` (Q6_K): max rel 4.935e-03, rms rel 5.898e-03, cos 0.999982715 |
| **PartialRoPE** | vs `mx.fast.rope` | ≤ 1e-5 | measured 1.151e-07; controls at 4.30e-01 / 9.37e-01 |

**Mode A exists precisely because Mode B's margin against the `alog` control
(2.3e-02) is thin.** Never let a Mode B gate stand in for a Mode A gate.

### The eps trap that must be documented, not "fixed"

HF's `l2norm` is `rsqrt(SUM(x²) + 1e-6)`; MLX routes the same call through
`mx.fast.rms_norm`, i.e. `rsqrt(MEAN(x²) + 1e-6)` — an effective eps **128×
larger**. Effect on the block output: **3.686e-04** (blk0), **6.240e-03** (blk1).
The blk1 figure is the same order as the Q8_K activation-quant budget, so **no
end-to-end cosine or logits gate can distinguish it.** The Haxe must implement the
HF sum form (which is what the passing run used). `verify_goldens.py` must keep
**both** rows in its table so a future reader does not "fix" the 3.686e-04 row by
changing the kernel.

---

## Steps

### Step 1 — the oracle, in-tree, extended

**Do:** Create `nue/tests/qwen35/`. Move `gguf.py` and `ref_gdn.py` verbatim.
Merge `gen_ssm_goldens.py` and the decode generator into one `gen_goldens.py`
that emits, into `goldens/`, for `LAYERS=[0,1]` and `TOKENS=[9707,11,4438,525,498,3351,30,358]`:
(a) the **prefill(8)** case — taps `x_norm, qkv, z, b, a, conv_raw, conv_out,
q_pre, k_pre, v, beta, g_log, g, q, k, core, state, gated, y, conv_state`;
(b) the **decode** case — `state`/`conv_state` after 7 tokens as *inputs*, the
t=7 step's taps as outputs, plus the F32 params `conv1d, A, dt_bias, ssm_norm`
so the kernel test needs no GGUF; (c) `embed`. **One** `manifest.json` carrying
`model`, `model_sha256_first1MB`, `tokens`, `cfg`, `layers`, `taps`, `checks`.
Add `ref_attn.py` (the full-attention block for blk.3, per THE MATH above) and
`ref_rope.py` (partial RoPE + the two controls). Extend `verify_goldens.py` to
**eight** controls and make it `sys.exit(1)` unless floor ≤ 1e-5 **and** every
control ≥ 1e-2. `goldens/README.md` copies the two sections
`nue/tests/bert/goldens/README.md` already carries: "Cross-validation (why the
golden is trustworthy)" and "Acceptance gates (calibrated to measured deltas)",
and states plainly that the sha256 covers only the first 1 MB — it is a
**directory check, not a content check**.

**Verify:** `$PY nue/tests/qwen35/verify_goldens.py` exits 0 and prints the
12-row table above (numpy-vs-mlx ≤ 1e-5 on blk0 and blk1). Controls
`repeat_interleave`, `ssm_a-as-A_log`, `conv-taps-reversed`, `ssm_norm-as-(1+w)`
reproduce at 3.809e-01 / 2.294e-02 / 3.109e-01 / 1.578e+00 (blk0). Four **new**
controls must be *measured, not assumed*, before their thresholds are written
into the gate: `flatsplit` (flat q|gate split), `fullrope`, `nogate`,
`interleaved_rope` — record what each actually moves and set the gate from that.
Manifest `checks` reproduce at 2.974e-07 / 1.444e-07. Assert the model hash
**before** comparing, not after.

**Files:** `nue/tests/qwen35/{gguf.py,ref_gdn.py,ref_attn.py,ref_rope.py,gen_goldens.py,verify_goldens.py}`,
`nue/tests/qwen35/goldens/{manifest.json,README.md,*.f32}`

---

### Step 2 — THE MILESTONE: pure-Haxe gated-delta-net decode step

**Do:** Add `nue/nue/transformer/GatedDeltaNet.hx`, `package nue.transformer`
(co-located beside `GQAttention.hx` — **not** a new package, not a new
directory). **Port `$SP/gdnprobe/Main.hx`'s kernel body; do not rewrite it from
the spec.** Instance methods only; no static→static (open bug:
`bugs_static_to_static_call_traps_in_import_module`, exit 133, silent). All eight
stages over raw `Usize` bases from `Tensor.data().raw()` with
`rayzor.Mem.loadF32`/`storeF32` — the `FlashDecode.hx` idiom. This is a
**correctness** choice, not a perf one: `prepare_binop` does
`from_raw_parts(a.data as *const f32, a.numel)` with no contiguity check
(`tensor.rs:2238-2262`), so `mul`/`add`/`scale` on a `slice` view silently
compute garbage; and leading-axis broadcast does not exist
(`tensor_binop_row_broadcast` requires `b.ndim == 1` and trailing-dim match,
`tensor.rs:2266+`), so a materialised `[32,128,128]` γ would cost 2 MB per layer
per token. Fusing decay+read+write+readout into one sweep of `S` also avoids
re-reading the 2 MiB state four times (8 MiB/layer/token → 192 MiB/token over 24
layers). Add `nue/tests/qwen35/Main.hx`: load fixtures with `File.getBytes` +
`Tensor.fromBytesF32` (`Tensor.hx:68`), compare each tap, print `ok`/`FAIL`,
`PASS` or `Sys.exit(1)` — the `nue/tests/flashbatch/Main.hx` convention.
`rayzor.toml` copies `nue/tests/q8/rayzor.toml`'s class-path trio;
`native-libs` = `librayzor_tensors.dylib` only (no KV/flash engine is touched).
Controls selected by `argv[1]`: `convtaps`, `interleave`, `alog`, `nodecay`,
`normplus1`.

**Verify:** Already measured with exactly this code, re-run at HEAD this session
(table at the top): all 10 taps ok, worst 2.353e-07 against a 1e-4 gate, `PASS`,
exit 0. The five controls each exit **1**, first-failing taps and magnitudes as
tabled. Assert exit codes explicitly — a `PASS` line with exit 133 is the failure
mode this codebase actually hits.

**Files:** `nue/nue/transformer/GatedDeltaNet.hx`, `nue/tests/qwen35/Main.hx`,
`nue/tests/qwen35/rayzor.toml`

---

### Step 3 — `GDNState` and the state-carry invariant

**Do:** Add `nue/nue/transformer/GDNState.hx`, `package nue.transformer`, modelled
on `Q8Cache.hx` (the working template for a guest-owned `Bytes`-backed per-layer
state object): recurrent `[32][128][128]` f32 = 2.00 MiB, conv ring `[3][8192]`
f32 = 96 KiB, plus `reset()` and `free()`. Change `GatedDeltaNet`'s entry point to
take `GDNState` as a **parameter**, never a field on the block —
`KVSession.hx:5-10` is the in-repo record of why module-owned cache state had to
be undone, and putting SSM state on the block would recreate that problem 24×
and block the multi-spoke direction.

**Verify:** `prefill(7)` then one decode step, final state and concatenated `y`
vs the one-shot `prefill(8)` golden, gate ≤ 1e-5 (reference residual 2.974e-07 /
1.444e-07). **Controls that must exit 1:** (a) `reset` the `GDNState` between the
two calls; (b) zero only the conv ring; (c) run the two steps in the wrong order.
This is the single most likely place a Haxe port is wrong, because step 2 only
ever runs the recurrence once.

**Files:** `nue/nue/transformer/GDNState.hx`, `nue/nue/transformer/GatedDeltaNet.hx`,
`nue/tests/qwen35/{gen_goldens.py,Main.hx}`

---

### Step 4 — real quantised weights; the whole SSM layer, `x_norm → y`

**Do:** **Use the existing public `GGUFLoader.readNamedTensors(path)`
(`GGUFLoader.hx:57-61`).** Do not add a new loader entry point. Build blk.0's and
blk.1's projections through the existing `nue.Linear`: `attn_qkv` (Q6_K),
`attn_gate` (Q4_K), `ssm_beta` and `ssm_alpha` (both `(32,4096)` **F32
matrices**, so a real F32 GEMV — not a vector op), and `ssm_out` (**Q8_0 on 12
layers, Q4_K on the other 12** — dispatch on the tensor's own dtype, never an
arch constant). Add the `attn_norm` RMSNorm on the way in and the `ssm_out`
projection on the way out. Split the conv output with
`slice(dim,a,b).reshape([...])` — `rayzor_tensor_slice` always returns a strided
view with `owns_data=false` (`tensor.rs:2111+`) while `reshape` detects
non-contiguity and materialises (`tensor.rs:1950`).

**Verify:** Mode B (`cos ≥ 0.9999`, rms-rel ≤ 2e-2) on taps `qkv, z, b, a` and on
`y`, against `blk0.*`/`blk1.*` — **both layers; one layer is not a pattern**
(blk1 is where the l2norm-eps class of error shows: 6.240e-03 vs blk0's
3.686e-04). Mode A (≤ 1e-4) still holds on every f32-only tap downstream of an
injected `conv_raw`. **Controls that must fail:** swap `ssm_alpha` and
`ssm_beta` (shape-identical `(32,4096)`, invisible to any shape check — only a
numeric tap catches it; fingerprint: alpha std 0.02456 vs beta std 0.00818); read
`attn_gate` from a full-attention layer (it does not exist there — must throw).
Assert **exit 0 and not 133**. Timing sanity: `File.getBytes` mmaps on unix
(`runtime/src/haxe_sys.rs:1703` calls `haxe_bytes_mmap_file` before falling back
to `fs::read`), so opening the 5.75 GB file costs ~0 and only blk.0/blk.1 pages
are touched.

**Files:** `nue/tests/qwen35/Main.hx`, `nue/nue/transformer/GatedDeltaNet.hx`

---

### Step 5 — PartialRoPE, isolated

**Do:** Add `nue/nue/transformer/PartialRoPE.hx`: pure Haxe over
`Tensor.data():Ptr<Float>` on `[seq, heads, 256]`, rotating **only** lanes
`[0,64)` with NeoX pairing `(i, i+32)`, leaving `[64,256)` untouched. Reuse
`Tensor.ropeCosTable(64, maxSeqLen, 1e7)` for the tables — no Rust change. Do
**not** try to express this with the existing kernel: `rope_impl` hardcodes
`half = head_dim/2`, gates on `cos_shape[1] != half` (returning **0 → a null
Tensor**, not a throw), and an identity-tail table still pairs at the wrong width.

**Verify:** Against `ref_rope.py` on one `[1,16,256]` tensor at several positions
(including `pos=0` and a position past the table, which the Rust kernel treats as
identity — decide and pin this model's behaviour explicitly). Gate ≤ 1e-5;
measured floor 1.151e-07. **Controls that must fail:** `fullrope` (rotate all
256) at ~9.4e-01; `interleaved_rope` (pair `2i, 2i+1`) at ~4.3e-01. Assert
lanes `[64:256]` are bit-identical to the input (measured 0.000e+00).
No SSM code is in the picture, which is why this step is early and cheap.

**Files:** `nue/nue/transformer/PartialRoPE.hx`, `nue/tests/qwen35/Main.hx`,
`nue/tests/qwen35/ref_rope.py`

---

### Step 6 — `GatedAttention` and the full-attention block

**Do:** Add `nue/nue/transformer/GatedAttention.hx` — a **new class**, not a
widening of `GQAttention` (hot class, append-only fields, and its
`numQHeads*headDim` width assumption breaks on a `(8192,4096)` q_proj;
`Tensor.reshape` returns **null** on a numel mismatch, and `GQAttention.hx:207`
feeds the result straight into `rope.apply` with no null check). Order exactly as
in THE MATH: q_proj → view `[seq,16,512]` → per-head `q = lanes[0:256]`,
`gate = lanes[256:512]` (**never** a flat `[0:4096]|[4096:8192]` split);
q_norm/k_norm over 256 **after** the split, gate untouched; PartialRoPE on q and
k; `kv.append`; naive bmm + causal mask + softmax + bmm (correctness first — no
`FlashDecode`, which assumes the standard shape); `ctx * sigmoid(gate)`; then
`attn_output`. Add `nue/nue/transformer/Qwen35AttnBlock.hx` wrapping it with
`attn_norm` / `post_attention_norm` / SwiGLU — note again there is **no**
`ffn_norm.weight` in this file. Mirror `LlamaBlock.forwardWith`'s in-place
`addInto` convention (`LlamaBlock.hx:111,131,146`) so a single `h.free()` at the
model level stays correct.

**Verify:** Inject `goldens/blk3.x_norm.f32`; gate taps `q, gate, k, v, q_normed,
k_normed, q_roped, k_roped, ctx, ctx_gated, y, block_out`. Mode A ≤ 1e-4 on the
f32-only taps; Mode B on the ones behind a quantised projection. **Controls that
must fail:** `flatsplit`, `fullrope`, `nogate`, `interleaved_rope` — thresholds
taken from step 1's measurements, not from this document.

**Files:** `nue/nue/transformer/GatedAttention.hx`,
`nue/nue/transformer/Qwen35AttnBlock.hx`, `nue/tests/qwen35/Main.hx`

---

### Step 7 — config, layer typing, metadata keys, and `TapDump`

**Do:** Append `"rope.dimension_count"` and `"nextn_predict_layers"` to
`ARCH_EXTRA_KEYS` (`GGUFLoader.hx:237-246`). This is append-only on a
`static final Array<String>` — safe: the field-ordering rule governs **instance**
fields. It changes `extras` for any arch that carries those keys, but `extras` is
only read by builders that ask for it. Add `nue/nue/arch/Qwen35Config.hx`,
`package nue.arch` (co-located with its future builder, per
`bugs_null_class_field_and_onearg_cast_xmodule`), deriving the real names from
the misleading keys and computing `keyDim=2048`, `convDim=8192`, `rep=2`,
`rotaryDim=64`, `bodyLayers = numLayers - nextnPredictLayers`. Layer typing as a
**computed** function, `classify(numLayers, nextn, interval)` → full-attention iff
`(i+1) % interval == 0`, not a hardcoded list. Read `extras` with
`exists()`/`get()` into a **bound local** only — never `Map.keys()` (SIGTRAPs,
exit 133, silent) and never `.get()` inline in a concat (prints the raw pointer;
`bugs_map_string_get_returns_raw_pointer_in_concat`). `Std.parseInt`'s `Null<Int>`
needs the `+ 0` unbox workaround as at `examples/llama-chat/Main.hx:66-74`.
Also add `nue/nue/TapDump.hx`: a standalone class (not a field on any hot class),
memoised `enabled()` Int gate on `NUE_TAP_DIR`, `write(name, t)` serialising
row-major LE f32 via `Tensor.data()` + `Mem.loadF32` into a `rayzor.Bytes`,
plus a `taps.txt` index — the same on-disk contract as
`nue/tests/bert/goldens/hidden/*.f32.bin`. Add `nue/tests/qwen35/compare_taps.py`.

**Verify:** A probe prints the derived config and asserts, each on its own line:
`bodyLayers==32`, `ssmLayers == [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]`
(the exact set enumerated from `ssm_a` presence), `attnLayers == [3,7,11,15,19,23,27,31]`,
`Hk==16, Hv==32, Dk==Dv==128, convDim==8192, rotaryDim==64, ropeBase==1e7,
headDim==256, nextn==1`. **Controls:** `classify(33,1,3)` must return
`[2,5,8,11,14,17,20,23,26,29]` — a hardcoded `/4` rule fails this; the naive
`i%4==0` rule yields `[0,4,…,28]` and must be rejected; a model lacking the
`ssm.*` keys (any Qwen2.5 GGUF in `~/models/qwen/`) must produce a **named
throw**, not a silent 0. For `TapDump`: dump an embedding row on an existing
working model and match `gguf.get_rows` at ≤ 1e-6; control 1 — compare against
the neighbouring row, must exceed 1e-2; control 2 — with `NUE_TAP_DIR` unset,
zero files and **no directory created**. Assert exit 0, not 133 (every Map
landmine here is silent).

**Files:** `nue/nue/loader/GGUFLoader.hx`, `nue/nue/arch/Qwen35Config.hx`,
`nue/nue/TapDump.hx`, `nue/tests/qwen35/compare_taps.py`,
`nue/examples/qwen35-plan/{Main.hx,rayzor.toml,run.sh}`

---

### Step 8 — the hybrid stack: session, model, arch — with the lifecycle controls

**Do:** Add `Qwen35Session` (`kv:Array<KVCache>` of 8, `ssm:Array<GDNState>` of
24, `pos:Int`), `Qwen35Model implements CausalLanguageModel`, and `Qwen35Arch`.
Stack shape: **parallel concrete arrays** `ssmBlocks:Array<GatedDeltaBlock>` +
`attnBlocks:Array<Qwen35AttnBlock>` plus `layerKind:Array<Int>` and
`layerSlot:Array<Int>`. No `Null<nue-class>` field (documented corruption record)
and no interface-typed array — `ArchRegistry.hx:82-88` records the JIT quirk
"interface-method dispatch fails when the receiver is typed as the interface",
which is why `register` takes the name as an explicit argument. **Do NOT touch
`ArchRegistry.withDefaults()` in this step**; the test injects its own registry
via the public `GGUFLoader.registry` field (`:63`, honoured at `:70` and `:184`).
Fix the four silent degradations: `cacheLen()` returns `session.pos`;
`cacheCapacity()` returns `meta.maxSeqLen`; `resetCache()` resets KV **and**
zeroes every `GDNState` **and** zeroes `pos`; `rewindCache()` **throws** with
`"qwen35"` in the message (the delta-rule state is destructively updated and has
no truncatable prefix). **Clamp `maxSeqLen` in the builder** — the file declares
262144, `loadWithTokenizer`'s `maxCtx` defaults to 0 and only clamps
`if (maxCtx > 0 && maxCtx < meta.maxSeqLen)` (`GGUFLoader.hx:148`, `:173-175`),
and the Q8 path allocates **eagerly in the constructor** (`KVCache.hx:76-81` →
`Q8Cache.hx:47-49`, unlike F32 which defers to `ensureF32`). Nothing in the
library clamps; only `examples/llama-chat/Main.hx`'s `DEFAULT_CTX=4096` saves the
shipped runner. Skip the MTP block **by index** (`bodyLayers`), never by
tensor-name presence — that saves 258.56 MB of load.

**Verify:** `GGUF=~/models/qwen/Qwen3.5-9B-Q4_K_M.gguf ./nue/examples/qwen35-stack-smoke/run.sh`
exits 0 and asserts: (a) the printed layer-visit trace is
`ssm,ssm,ssm,attn,…` matching step 7's `attnLayers`; (b) allocated SSM state
== `24 * 2195456` == 52,690,944 B and exactly **8** `KVCache`s exist;
(c) `pos == 8` after the prompt, 9 after one decode step; (d) **the poison/reset
control** — write a sentinel into `ssm[0]`'s recurrent state, call
`resetCache()`, read back 0.0; with the SSM half of `resetCache` commented out
the sentinel survives and the run exits 1; (e) `rewindCache(3)` throws and the
message contains `"qwen35"`; (f) a session with the wrong layer count throws;
(g) `weights.exists("blk.32.attn_q.weight")` is true but the taken-name census
shows blk.32 was never read; (h) forward the same prompt twice **without**
`resetCache` and assert the second result **differs** — proving the reset is
load-bearing. Grep the run for `LLVM upgrade failed` (must be absent) per
`bugs_restored_modules_lose_the_llvm_tier`; assert exit 0 and not 133/139.

**Files:** `nue/nue/transformer/{Qwen35Session.hx,GatedDeltaBlock.hx}`,
`nue/nue/arch/{Qwen35Model.hx,Qwen35Arch.hx}`,
`nue/examples/qwen35-stack-smoke/{Main.hx,rayzor.toml,run.sh}`

---

### Step 9 — logits gate, registry registration, chat runner

**Do:** Extend `ref_model.py` to drive all 32 body layers to final logits for the
8 pinned token ids, emitting `h_after_0/3/15/31`, `logits.f32` and a top-5 id
list. Instrument `Qwen35Model`'s block loop with `TapDump.write`. **Only now**
register `"qwen35"` in `ArchRegistry.withDefaults()`. Add
`nue/examples/qwen35-chat/` (a copy of `llama-chat/Main.hx` casting to
`Qwen35Model` — `GenerationLoop` and `Conversation` are typed to the
`CausalLanguageModel` interface, so nothing in the sampling stack changes; note
`GGUFLoader.applyEngines` is gated on `architecture == "llama" || "qwen2"`
(`:83`), so the prefill-graph/decode-warm wiring is inert for qwen35 and that is
correct for now). Add the qwen35 test to `run_haxe_tests.sh` beside the existing
`nue_smoke_llama_chat` block (`:118-142`) — note that harness currently runs
**no** `nue/tests/*` at all.

**Verify:** Top-1 id equals the golden top-1; top-5 ids equal the golden top-5 as
a **set**; final-row logits cosine ≥ 0.999; the per-layer cosine curve against
`h_after_0/3/15/31` is monotone with layer-31 ≥ 0.999. **Controls that must break
top-1:** `nomtpskip` (run all 33 blocks); `interval3`; `noresetssm`.
**Diagnostic built into the gate:** if the cosine curve steps down *only* at
layers 3, 7, 11, … the cause is `GGUFLoader.hx:388-391` re-encoding Q5_K to
Q4_K_M — all 16 Q5_K tensors are `attn_k.weight` and `attn_output.weight` on
exactly those 8 layers and SSM layers carry **zero** Q5_K, so the diagnosis is
decisive; the fix is a true Q5_K decode path before the gate is tightened.
Finally run the chat example with an explicit `maxCtx` and report decode tok/s —
after grepping for `LLVM upgrade failed`.

**Files:** `nue/nue/arch/{Qwen35Arch.hx,Qwen35Model.hx,ArchRegistry.hx}`,
`nue/tests/qwen35/ref_model.py`,
`nue/examples/qwen35-logits-check/{Main.hx,run.sh}`,
`nue/examples/qwen35-chat/{Main.hx,rayzor.toml,run.sh}`, `run_haxe_tests.sh`

---

### Step 10 — tighten, separately, moving no tap

**Do:** Three levers, each landed and measured **on its own commit**.
(1) Parallelise the recurrence. For a fixed `(hv, dv)` the entire step touches
only **column `dv`** of `S[hv]`, so there are `Hv*Dv = 4096` fully independent
lanes per layer, each a `Dk=128` f32 vector (512 B); `q[hv]`/`k[hv]` are shared
across a head's 128 lanes. Dispatch over `SpinPool.parallelRows` and make the
**token loop the inner loop** so the lane's 512 B stays in registers — that buys
the arithmetic intensity HF's chunked/WY prefill buys, without implementing a
second algorithm. (2) Q8 KV + `FlashDecode` on the 8 attention layers —
`headDim=256` passes `KVCache.hx:74`'s `canQ8 = useQ8 && (headDim % 32 == 0)`.
(3) Batched prefill for the conv.

**Verify:** Every gate from steps 2–9 re-runs **unchanged** and still passes at
the same thresholds. **A tightening step that moves any tap is a miscompile, not
a win.** Then the A/B per the standing protocol: `NUE_MATMUL_WORKERS=1` +
`cpu_core/cycles/`, three runs per arm, stddev > ~1 means contamination. Print a
one-line census of which recurrence kernel actually ran **before** trusting any
number — the `perf_haxe_batched_prefill_attention` scar is a win that could not
be shown to have taken the path it claimed.

**Files:** `nue/nue/transformer/{GatedDeltaNet.hx,GatedAttention.hx,GDNState.hx}`

---

## Must not do

- **Do NOT add a new public per-tensor decode entry point to `GGUFLoader`.**
  `public function readNamedTensors(path)` already exists at `GGUFLoader.hx:57-61`.
  Adding surface on top of the static→static exit-133 trap for a capability that
  ships is pure risk.
- **Do NOT use `Tensor.expandKvHeadsAxis1` on the SSM path.** It computes
  `kvh = qh / repeats` (`tensor.rs:1897`) — repeat-interleave. GGUF stores V-rows
  **tiled** (`qwen_conv.py:455-470`), so the k-head is `hv % Hk`. The wrong choice
  costs 1.09e+00 at the `q` tap and produces fluent text.
- **Do NOT re-exponentiate `ssm_a`.** It already holds `-exp(A_log)`
  (`qwen_conv.py:388-389`; all 768 values negative). `g = ssm_a * softplus(a + dt_bias)`.
  Porting HF's line verbatim moves `gamma` by 7.14e-01 and the block output by
  only 2.29e-02 — the second number is why a logits-level gate misses it.
- **Do NOT add `+1` to `ssm_norm.weight`.** The converter folds `+1` into every
  `norm.weight` **except** `linear_attn.norm.weight` (`qwen_conv.py:394-395`).
  Measured: `ssm_norm` mean 0.88117 vs `attn_norm` mean 1.03306.
- **Do NOT use `rmsNorm`'s mean form for the q/k l2norm.** HF is
  `rsqrt(SUM(x²)+1e-6)`; the mean form is an effective eps 128× larger and moves
  blk1's output by 6.240e-03 — inside the quantisation budget, so no end-to-end
  gate can see it. Gate it at the tap with its own named control.
- **Do NOT split `attn_q.weight` as `[0:4096] | [4096:8192]`.** It is 16 heads ×
  (256 q | 256 gate), interleaved per head
  (`mlx_lm/models/qwen3_next.py:129-133`). A flat split is a silent miscompile.
- **Do NOT reuse `rope`/`ropeNeox` for the attention layers.** `rope_impl`
  hardcodes `half = head_dim/2` and returns **0 → null** on a table-width
  mismatch (`tensor.rs:3072`, `:3081`). Write `PartialRoPE` in Haxe.
- **Do NOT do elementwise Tensor ops on a `slice` view.** `prepare_binop` reads
  `numel` linearly from `a.data` with no contiguity check (`tensor.rs:2238-2262`);
  `rayzor_tensor_slice` always returns a strided view (`tensor.rs:2111+`). Use
  `slice(...).reshape(...)` (reshape materialises, `tensor.rs:1950`) or raw
  `Mem` loads with explicit offsets.
- **Do NOT put SSM state on the block.** `KVSession.hx:5-10` records why
  module-owned cache state had to be undone. State is a parameter, always.
- **Do NOT add a `Null<nue-defined-class>` instance field**, and do NOT widen
  `LlamaModel`/`LlamaBlock`. `LlamaBlock.attn` is a non-nullable `GQAttention`
  (`:17`) and `LlamaModel.blocks` is `Array<LlamaBlock>` (`:52`) — the element
  **type** is what has to change, which is exactly what append-only cannot do.
- **Do NOT register `"qwen35"` in `ArchRegistry.withDefaults()` before step 9.**
  Inject a registry via the public `GGUFLoader.registry` field. `LlamaArch.validate()`
  **passes** this model (`256*16 == 4096`) and would then die on the nonexistent
  `ffn_norm.weight` (`LlamaArch.hx:287`) — a loud failure today is a feature.
- **Do NOT skip block 32 by tensor name.** It is a complete decoder layer plus
  four `nextn.*` tensors, 258.56 MB, and `attn_q.weight` is present on **9**
  blocks. Skip by index: `blockCount - nextn_predict_layers`.
- **Do NOT loop `0...meta.numLayers`.** That is 33.
- **Do NOT let `maxSeqLen` default.** The file declares 262144; the Q8 cache
  allocates eagerly in its constructor. The builder must clamp itself.
- **Do NOT use `Map.keys()`** (SIGTRAPs, exit 133, silent) and do NOT use
  `Map<String,String>.get()` inline in a concat (prints the raw pointer — it
  fabricates address-shaped values that read as plausible config).
- **Do NOT gate on `y`, on logits, or on block output alone.** Gate per tap. The
  weakest control is 2.294e-02 at block output; never loosen past ~1e-3.
- **Do NOT shrink the golden below 32 v-heads.** With `hv` in `0..3`, tile and
  repeat-interleave become nearly indistinguishable and the strongest control is
  silently weakened. If size must come down, the only safe trim is dropping
  `conv_raw` (redundant with `conv_out`).
- **Do NOT lean on the perplexity gate.** `bench/eval` SIGABRTs (exit 134,
  SILENT) entering chunk 2 and `gate.sh` defaults to 6 chunks
  (`bugs_bench_eval_aborts_on_second_chunk`). This plan's acceptance is entirely
  layer-wise and does not need it — keep it that way.
- **Do NOT install torch, and do NOT download anything.** `mlx` 0.32.1 +
  `mlx_lm` 0.31.3 already provide the independent second implementation.
- **Do NOT batch these steps into one commit.** The recorded E0100/SIGSEGV/133
  failures are non-local and cannot be bisected by inspection.
- **Do NOT reach for `RAYZOR_DUMP_FN_PTRS=1`** to diagnose an exit 133 — that flag
  makes a clean build SIGILL (132), so the diagnostic perturbs the run.

## Open risks

- **`rewindCache` is unimplementable and it is in the interface.**
  `nue/nue/CausalLanguageModel.hx:55` declares it; `SpeculativeGenerationLoop`
  builds on `cacheLen()` + rewind. Throwing is the honest answer, but it removes
  speculative decoding for this architecture entirely and turns a defensive call
  into a hard failure. Decide this **before** writing `Qwen35Model` — the
  generation loops dispatch through the interface.
- **The Q5_K re-encode puts an irreducible floor under step 9** that no correct
  code removes. The plan makes the diagnosis decisive but the fix (a true Q5_K
  decode path) is unscoped work that may have to land between steps 6 and 9.
- **The `ssm_alpha` / `ssm_beta` assignment is inference, not proof from the
  file.** Both are `(32,4096)` F32. The oracle agreeing at 9.7e-07 with the
  alphabetical assignment *is* strong evidence, but only because the whole block
  matched; nothing in the file labels which drives which nonlinearity. Keep the
  swap control permanently.
- **Step 2's kernel is scalar and single-threaded** over a 2 MiB state per head,
  24 layers. The first tok/s number in step 9 will be bad, and the temptation to
  fold step 10 into step 9 to make it look better is exactly the move that makes
  the A/B unreadable. Resist it.
- **The 50.25 MiB SSM floor is per session and constant in context.** Below ~268
  tokens (F32) / ~817 tokens (Q8) the hybrid state is *larger* than the KV it
  replaces, and N concurrent spokes multiply it. This cuts against the hub/spoke
  direction at short contexts; the memory win appears only at long ones.
- **Adding two keys to `ARCH_EXTRA_KEYS` changes `extras` for every arch that
  carries them.** Inert because only builders that ask read `extras`, but
  anything that hashes or serialises `ModelMetadata` sees a different map.
- **The goldens pin an absolute model path and a sha256 of only the first 1 MB.**
  That covers the header and tensor directory but **not one weight**. A
  re-quantised file with an identical header passes the hash and silently
  produces wrong goldens. Assert the hash before comparing and say plainly in the
  README that it is a directory check.
- **ID-space perturbation.** This work adds ~8 classes and many instance fields
  across `nue.transformer` and `nue.arch`. The recorded E0100/SIGSEGV failures
  are declaration-change-driven and non-local. Mitigated by co-locating in
  existing packages, appending fields, and staging — but proven only by running.
  Probes if something breaks: `RAYZOR_ALLOC_DEBUG=1`, `RAYZOR_CTOR_DEBUG`,
  `RAYZOR_FIELD_DEBUG`.
- **Verification runs on arm64 macOS and the id-space hazard is host-dependent.**
  A green Mac run does not exonerate new classes; the new-module SIGSEGV record is
  x86-only. Until an x86 build+run happens, the claim is "clean on arm64 Mac".
- **Thin evidence, flagged as such:** (a) the four new attention/rope controls
  (`flatsplit`, `fullrope`, `nogate`, `interleaved_rope`) have **not** been
  measured end-to-end on a real full-attention layer — only the rope pairing was,
  in isolation, this session; their gate thresholds must be set from step 1's
  measurements, not from this document. (b) The whole-body logits path
  (`ref_model.py`) is unbuilt. (c) The MTP block has **no** validation target —
  transformers 5.15.1 contains no MTP/nextn implementation at all — so it can
  only be skipped and the skip asserted, never validated.
