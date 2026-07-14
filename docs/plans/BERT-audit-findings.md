# BERT encoder audit — findings & Phase 1 worklist

Multi-agent audit (8 components, 32 findings, each double-verified by two refutation lenses)
of the existing untested BERT code against canonical HF BERT + the **real** all-MiniLM-L6-v2
GGUF. Ground truth from `~/models/minilm/all-MiniLM-L6-v2-ggml-model-f16.gguf`:

```
general.architecture = bert          bert.attention.layer_norm_epsilon = 1e-12
bert.block_count = 6                  bert.pooling_type = 1   (1 = MEAN)
bert.embedding_length = 384           bert.attention.causal = False
bert.feed_forward_length = 1536       tokenizer.ggml.model = bert   (WordPiece)
bert.attention.head_count = 12
per-layer norm tensors: blk.L.attn_output_norm.{weight,bias}, blk.L.layer_output_norm.{weight,bias}
non-block tensors: token_embd[_norm], position_embd, token_types
```
Note the per-layer norm names are `attn_output_norm` / `layer_output_norm` — NOT the
`attn_norm` / `ffn_norm` BertArch currently expects. These names are themselves post-norm
terminology (norm applied to the sublayer *output*).

## BLOCKERS (embeddings wrong or load crashes) — all survived verification

| # | id | file:line | defect | fix |
|---|---|---|---|---|
| 1 | transformerblock-prenorm-attn | TransformerBlock.hx:68 | attn sublayer is **pre-norm**; BERT is post-norm | run attn on x, add residual, THEN norm the sum |
| 2 | transformerblock-prenorm-ffn | TransformerBlock.hx:78 | ffn sublayer is **pre-norm** | run ffn on a, add residual, THEN norm the sum |
| 3 | bertmodel-no-pooling / pooling-head-missing | BertModel.hx:82, BertArch.hx:88 | no **mean+L2 pooling** head → no sentence vector | add masked-mean + L2 pooling in BertModel |
| 4 | wordpiece-tokenizer-unsupported | GGUFTokenizer.hx:39 | WordPiece vocab built as a **merge-less BPE** → garbage tokens | branch modelType=="bert" → WordPiece (greedy longest-prefix + `##`) |
| 5 | return-type-pinned-bpetokenizer | GGUFLoader.hx:76 | tokenizer return type hard-typed `BPETokenizer` | widen to `Tokenizer` interface |
| 6 | layernorm-key-mismatch-gguf | BertArch.hx:106 | expects `attn_norm`/`ffn_norm`; GGUF has `attn_output_norm`/`layer_output_norm` → hard-throw | GGUF loader rename table OR BertArch fallback names |
| 7 | embed-norm-optional | BertArch.hx:79 | required post-embed LayerNorm uses `takeOptionalNorm` → can vanish | mandatory `takeNorm` for bert |
| 8 | normeps-rms-only-key | GGUFLoader.hx:188 | eps read only from RMS key; bert stores `attention.layer_norm_epsilon` | read bert key first, default 1e-12 for bert |

## MAJORS (measurable error) — kept

| id | file:line | defect | fix |
|---|---|---|---|
| geluffn-tanh-not-erf | GeluFFN.hx:30 | uses tanh GELU (`gelu_new`); MiniLM needs **exact erf** GELU (config hidden_act="gelu") | add erf-GELU kernel, route encoder FFN through it |
| mha/attn-mask-not-wired | MHA.hx:56, BertArch.hx:104, BertModel.hx:77 | attention_mask never applied → padded keys attended | thread additive `(1-mask)*-inf` into scores; harmless for single unpadded seq |
| pooling-type-not-read | GGUFLoader.hx:173 | `bert.pooling_type` not read into metadata | read it; add `poolingType` to ModelMetadata |

## MINORS — kept (do opportunistically)
- `int-buffer-dtype-crashes-decode` (SafetensorsLoader.hx:127): non-F32/F16/BF16 tensors throw
  at decode (position_ids/token_type_ids int buffers) → skip integer buffers.
- `bert-special-tokens-and-token-type-ignored` (GGUFTokenizer.hx:69): register
  `[CLS]/[SEP]/[PAD]/[UNK]/[MASK]` for bert; consult token_type list.
- eps fallback defaults (LayerNorm.hx:24, SafetensorsLoader.hx:245): 1e-5 RMS default; prefer
  1e-12 for bert (small numeric impact when var~1).
- docstrings (TransformerBlock.hx:22, BertModel.hx:11, BertArch.hx:23) claim pre-norm — correct
  to post-norm once code is fixed.

## DROPPED (both refutation lenses refuted — verified non-issues)
- `bertmodel-tokentype-dropped-when-null`: all-MiniLM GGUF **has** `token_types.weight`, so
  segmentEmbed is non-null and token_type[0] IS added. (Golden includes token_type_ids=0.)
- `bertmodel-embednorm-skippable`: norm is wired for the real model.
- `vocabsize-bogus-fallback-key`: fallback path not hit (vocab_size present).
- `ropebase-meaningless-for-bert`: bert builder never consumes ropeBase.
- `layernorm-default-eps-1e5-not-1e12`: loader passes the real eps; default never used in practice.

## CONFIRMED-CORRECT (audited, no defect — do NOT touch)
- MHA scale (1/√headDim), softmax axis, reshape/permute, clone/move discipline.
- LayerNorm kernel: population variance, eps inside sqrt, γx+β.
- SafetensorsLoader: BERT name mapping is correct, the two-LayerNorm trap is NOT tripped, and
  HF `[out,in]` weight layout matches nue Linear's matmulT → **no transpose needed**.
- GGUF generic key reads (embedding_length/block_count/head_count/feed_forward/context_length).
- Embedding lookup + three-embedding sum + post-embedding LayerNorm ordering.

## Phase 1 critical path (ordered)
1. **Post-norm** TransformerBlock (blocker 1,2) — biggest single correctness fix.
2. **erf GELU** kernel (major) — needed for f32 golden ≥0.999.
3. **GGUF bert wiring** (blockers 6,7,8; major pooling-type): norm-name rename table,
   `attention.layer_norm_epsilon`, `pooling_type`, mandatory embed norm.
4. **WordPiece tokenizer** (blockers 4,5) — greedy longest-prefix + `##`; widen return type.
5. **Mean+L2 pooling** head (blocker 3) with mask.
6. **Attention mask** threading (major) — required for Phase 2 batching; single-seq no-op now.
7. Minors (int-buffer skip, special tokens) as encountered.

Verify each against `nue/tests/bert/goldens/` — tokens exact, then hidden_0 per-layer, then
cosine ≥0.999. Expect cross-module resolution-class compiler findings when first exercised.
