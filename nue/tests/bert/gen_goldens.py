#!/usr/bin/env python3.13
"""Generate BERT golden fixtures for all-MiniLM-L6-v2 — torch-free.

Reference paths:
  - tokenization: HF `tokenizers` lib loading the model's own tokenizer.json (authoritative WordPiece)
  - embeddings:   ONNX model.onnx via onnxruntime -> last_hidden_state -> masked mean pool -> L2
                  (reproduces sentence-transformers exactly: pooling=mean, then normalize)

Outputs -> nue/tests/bert/goldens/
"""
import json, os, glob, struct, sys
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))[0]
OUT = "/Users/amaterasu/Vibranium/rayzor/nue/tests/bert/goldens"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/hidden", exist_ok=True)

SENTENCES = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "BERT is a bidirectional encoder representation from transformers.",
    "Tokenization handles subwords like unbelievability and antidisestablishmentarianism.",
    "café naïve résumé façade",                       # accent stripping (uncased)
    "COVID-19 spread rapidly across the globe in 2020.",
    "Please email me at foo.bar@example.com when ready.",
    "Visit https://example.com/path?q=1&lang=en for details.",
    "3.14159 approximates pi, and 2 to the 10th equals 1024.",
    "Don't you think it can't and won't work out?",   # contractions
    "SUPERCALIFRAGILISTICEXPIALIDOCIOUS is a very long word.",  # all-caps rare
    "a",                                              # single char
    "The",                                            # single common word
    "Multiple   internal    spaces   collapse.",      # whitespace
    "Hello, world! How are you doing today? I'm fine, thanks.",
    "机器学习 and artificial intelligence reshape society.",  # CJK + latin
    "The years 1999, 2000, and 2024 were notable.",
    "Rock 'n' roll music played all through the night.",
    "Anthropic builds reliable and steerable AI systems for everyone.",
    "Machine learning transforms industries and workflows worldwide.",
    "Deep neural networks learn hierarchical feature representations from raw data across many layers of nonlinear transformation and abstraction.",  # long
    "A dog runs happily through the green park.",
    "A canine sprints across the grassy field.",      # semantic pair with prev
    "The stock market crashed on Monday morning.",
    "Equity prices plummeted at the start of the week.",  # semantic pair
    "I love programming in a fast systems language.",
    "Writing efficient low-level code is deeply satisfying.",  # semantic pair
    "Quantum computers exploit superposition and entanglement.",
    "The recipe calls for two cups of flour and one egg.",
    "Bake at 350 degrees Fahrenheit for twenty-five minutes.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The quick brown fox jumps over the lazy dog.",   # exact dup of #1 (determinism check)
]
assert len(SENTENCES) == 32, len(SENTENCES)

# --- tokenizer reference (authoritative WordPiece) ---
tok = Tokenizer.from_file(f"{SNAP}/tokenizer.json")
tok.no_padding()       # natural length — our Haxe WordPiece emits raw ids, no [PAD] fill
tok.no_truncation()    # never truncate the goldens
tok_out = []
for i, s in enumerate(SENTENCES):
    enc = tok.encode(s)
    tok_out.append({
        "idx": i, "text": s,
        "ids": enc.ids, "tokens": enc.tokens,
        "type_ids": enc.type_ids, "attention_mask": enc.attention_mask,
        "n": len(enc.ids),
    })
json.dump(tok_out, open(f"{OUT}/tokens.json", "w"), ensure_ascii=False, indent=1)
print(f"tokens.json: 32 sentences, lengths {min(t['n'] for t in tok_out)}..{max(t['n'] for t in tok_out)}")

# --- ONNX reference embeddings ---
sess = ort.InferenceSession(f"{SNAP}/onnx/model.onnx", providers=["CPUExecutionProvider"])
in_names = {i.name for i in sess.get_inputs()}
out_names = [o.name for o in sess.get_outputs()]
print("onnx inputs:", sorted(in_names), "outputs:", out_names)

DIM = 384
emb = np.zeros((32, DIM), dtype=np.float32)
for t in tok_out:
    ids = np.array([t["ids"]], dtype=np.int64)
    mask = np.array([t["attention_mask"]], dtype=np.int64)
    tt = np.array([t["type_ids"]], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in in_names:
        feed["token_type_ids"] = tt
    res = sess.run(None, feed)
    hidden = res[0][0]  # [seq, 384] last_hidden_state, batch=1 (no padding)
    if t["idx"] < 3:  # dump raw hidden for first 3 (layer-localization debugging)
        hidden.astype(np.float32).tofile(f"{OUT}/hidden/hidden_{t['idx']}.f32.bin")
    m = np.array(t["attention_mask"], dtype=np.float32)[:, None]
    pooled = (hidden * m).sum(0) / m.sum()          # masked mean pool
    pooled = pooled / np.linalg.norm(pooled)        # L2 normalize
    emb[t["idx"]] = pooled.astype(np.float32)

emb.tofile(f"{OUT}/embeddings.f32.bin")
json.dump({
    "rows": 32, "dim": DIM, "dtype": "f32", "layout": "row-major-le",
    "source": "onnxruntime all-MiniLM-L6-v2 model.onnx",
    "pooling": "masked-mean", "normalize": "l2", "order": "matches sentences.txt line order",
}, open(f"{OUT}/embeddings.meta.json", "w"), indent=1)

# self-consistency: exact-duplicate sentences (#1 idx1 vs #31 idx31) must match ~exactly
dup_cos = float(emb[1] @ emb[31])
print(f"dup cosine (#1 vs #31, identical text): {dup_cos:.6f}  (must be ~1.0)")

# semantic sanity: paired sentences should out-cosine random pairs
pairs = [(21, 22), (23, 24), (25, 26)]
for a, b in pairs:
    print(f"semantic pair cos[{a},{b}] = {float(emb[a]@emb[b]):.4f}")
cos = emb @ emb.T
json.dump({"cosine": cos.round(5).tolist()}, open(f"{OUT}/cosine_matrix.json", "w"))

# provenance
cfg = json.load(open(f"{SNAP}/config.json"))
json.dump({
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "arch": "bert", "hidden_size": cfg["hidden_size"], "num_layers": cfg["num_hidden_layers"],
    "num_heads": cfg["num_attention_heads"], "intermediate": cfg["intermediate_size"],
    "vocab": cfg["vocab_size"], "max_pos": cfg["max_position_embeddings"],
    "layer_norm_eps": cfg["layer_norm_eps"], "hidden_act": cfg["hidden_act"],
    "pooling": "mean", "normalize": "l2",
    "gates": {"tokenizer_ids": "exact", "cosine_f32": 0.999, "cosine_f16": 0.99, "cosine_int8": 0.985},
}, open(f"{OUT}/model.json", "w"), indent=1)

with open(f"{OUT}/sentences.txt", "w") as f:
    for s in SENTENCES:
        f.write(s.replace("\n", " ") + "\n")

print(f"\nwrote fixtures to {OUT}")
for f in sorted(os.listdir(OUT)):
    p = f"{OUT}/{f}"
    if os.path.isfile(p):
        print(f"  {f:28s} {os.path.getsize(p):>8d} B")
