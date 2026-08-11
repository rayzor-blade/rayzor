# BERT golden fixtures — all-MiniLM-L6-v2

Acceptance reference for Nue's BERT encoder.
Model: `sentence-transformers/all-MiniLM-L6-v2` — 6-layer BERT, hidden 384, 12 heads, head-dim
32, ffn 1536, vocab 30522, max-pos 512, LayerNorm eps 1e-12, **exact erf GELU**, POST-norm,
learned absolute positions. Sentence embedding = **masked mean pool + L2 normalize** (no [CLS]
pooler).

## What our encoder must match
- **`tokens.json`** — 32 sentences → WordPiece token ids (EXACT match required). Reference =
  the model's own `tokenizer.json` via HF `tokenizers`. Includes `[CLS]`/`[SEP]`, `##`
  continuations, type_ids, attention_mask, natural length (no padding).
- **`embeddings.f32.bin`** — `[32, 384]` row-major LE f32, pooled + L2-normalized sentence
  embeddings. Reference = ONNX `model.onnx` via onnxruntime (reproduces sentence-transformers).
- **`hidden/hidden_{0,1,2}.f32.bin`** — raw `last_hidden_state` `[seq, 384]` for the first 3
  sentences, for **per-layer localization** when debugging a divergence.
- **`embeddings_gguf_f32.json`** — the same embeddings from llama.cpp on the F32 GGUF (the
  GGUF-loader path reference).
- **`cosine_matrix.json`** — `[32,32]` cosine sims (self-consistency + semantic sanity).
- **`model.json`** — config + acceptance gates + provenance.

## Cross-validation (why the golden is trustworthy)
ONNX (safetensors) and llama.cpp (GGUF-f32) agree to **cosine 1.00000 on all 32 rows** — two
independent implementations, same answer. Quantization deltas vs the f32 reference are measured,
not guessed: **f16 = 1.00000**, **q8_0 worst = 0.99969**.

## Acceptance gates (calibrated to measured deltas + headroom for our GEMM order)
| path | gate (cosine vs golden) |
|---|---|
| token ids | exact |
| f32 / f16 | ≥ 0.999 |
| q8_0 | ≥ 0.995 |

A pre-norm/post-norm swap or wrong LayerNorm eps tanks cosine below 0.9 — these gates catch
structural bugs while tolerating floating-point reassociation.

## Regenerate
GGUFs live outside the repo in `~/models/minilm/` (f16, q8_0, f32 — `hf download` from
`second-state/All-MiniLM-L6-v2-Embedding-GGUF` and `leliuga/all-MiniLM-L6-v2-GGUF`).
```
python3.13 nue/tests/bert/gen_goldens.py     # needs: tokenizers, numpy, onnxruntime (no torch)
```
The 32-sentence set stresses WordPiece: accents (strip), subwords, numbers, URLs, contractions,
CJK, all-caps, whitespace, plus semantic pairs and an exact duplicate (#1==#31, determinism).
