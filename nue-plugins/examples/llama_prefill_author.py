# Author the FULL Llama prefill graph (all blocks) as one CoreML mlprogram
# per sequence bucket, weights baked from the model's own GGUF (Q4_K
# dequantized -> fp16). The 2-block probe measured this ~15x over the per-op
# Q4 pipeline on CPU (ANE rejects llama dims — CPU_ONLY is the engine).
#
#   input : h   [S, hidden] f32   (post-embedding hidden states; pad rows
#                                  with zeros — causal masking keeps padded
#                                  positions from touching real rows)
#   output: out [S, hidden] f32   (final hidden, PRE output_norm)
#           k{i}, v{i} [S, kv_heads, head_dim] f32 per layer — K POST-rope,
#           exactly what KvCacheQ8.append() consumes on the CPU decode side.
#
# ROPE IS INTERLEAVED (pairs x[2i], x[2i+1], freq i = base^(-2i/d)) — the
# GGUF llama convention rayzor's decode kernel implements
# (rayzor-tensors/src/tensor.rs "GGUF Llama models use the *interleaved*
# RoPE convention"). The 2-block probe used NEOX half-rotation, which is
# self-consistent but produces K caches the CPU decode CANNOT read. cos/sin
# tables and the causal mask are baked per bucket.
#
# Usage:
#   mlvenv/bin/python llama_prefill_author.py <llama.gguf> <outdir> [buckets...]
#   xcrun coremlc compile <outdir>/<stem>.prefill_s{S}.mlpackage <outdir>
#
# Buckets are fixed-shape (one artifact each, ~2.5 GB fp16 for 1B): CoreML
# enumerated shapes only allow ONE flexible input and the baked tables are
# shape-dependent. Palettization (4-bit LUT, ~700 MB) is the follow-up once
# the coherence gate passes on fp16.

import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from gguf import GGUFReader
from gguf import quants as gq

HIDDEN = HEADS = KV_HEADS = HEAD_DIM = FFN = LAYERS = 0
EPS = 1e-5
ROPE_BASE = 500000.0
STEM = ""


def meta_int(r, key):
    f = r.fields[key]
    return int(f.parts[f.data[0]][0])


def meta_float(r, key, default):
    if key not in r.fields:
        return default
    f = r.fields[key]
    return float(f.parts[f.data[0]][0])


def load_model(gguf_path):
    r = GGUFReader(gguf_path)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    assert arch == "llama", f"expected a llama gguf, got {arch!r}"
    dims = {
        "hidden": meta_int(r, "llama.embedding_length"),
        "layers": meta_int(r, "llama.block_count"),
        "heads": meta_int(r, "llama.attention.head_count"),
        "kv_heads": meta_int(r, "llama.attention.head_count_kv"),
        "ffn": meta_int(r, "llama.feed_forward_length"),
        "rope_base": meta_float(r, "llama.rope.freq_base", 10000.0),
        "eps": meta_float(r, "llama.attention.layer_norm_rms_epsilon", 1e-5),
    }
    want = set()
    for i in range(dims["layers"]):
        p = f"blk.{i}."
        for n in (
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ):
            want.add(p + n)
    w = {}
    for t in r.tensors:
        if t.name not in want:
            continue
        data = gq.dequantize(t.data, t.tensor_type)
        w[t.name] = np.asarray(data, dtype=np.float32).reshape(
            tuple(int(d) for d in reversed(t.shape))
        )
    missing = [n for n in want if n not in w]
    assert not missing, f"missing tensors: {sorted(missing)[:4]}"
    return w, dims


def rope_tables(S):
    # freq index i belongs to the INTERLEAVED pair (x[2i], x[2i+1]).
    half = HEAD_DIM // 2
    inv = 1.0 / (ROPE_BASE ** (2.0 * np.arange(half, dtype=np.float64) / HEAD_DIM))
    ang = np.outer(np.arange(S, dtype=np.float64), inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


# ---------- numpy reference (interleaved rope) -------------------------------

def np_rms(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g


def np_rope_interleaved(x, cos, sin):
    # x: [H, S, D] -> pairs on the last axis.
    H, S, D = x.shape
    p = x.reshape(H, S, D // 2, 2)
    e, o = p[..., 0], p[..., 1]
    re = e * cos - o * sin
    im = e * sin + o * cos
    return np.stack([re, im], -1).reshape(H, S, D)


def np_forward(h, w, cos, sin, mask):
    S = h.shape[0]
    scale = 1.0 / np.sqrt(HEAD_DIM)
    ks, vs = [], []
    for i in range(LAYERS):
        p = f"blk.{i}."
        x = np_rms(h, w[p + "attn_norm.weight"])
        q = (x @ w[p + "attn_q.weight"].T).reshape(S, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = (x @ w[p + "attn_k.weight"].T).reshape(S, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = (x @ w[p + "attn_v.weight"].T).reshape(S, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
        q = np_rope_interleaved(q, cos, sin)
        k = np_rope_interleaved(k, cos, sin)
        ks.append(k.transpose(1, 0, 2).copy())
        vs.append(v.transpose(1, 0, 2).copy())
        kr = np.repeat(k, HEADS // KV_HEADS, 0)
        vr = np.repeat(v, HEADS // KV_HEADS, 0)
        s = q @ kr.transpose(0, 2, 1) * scale + mask[None]
        a = np.exp(s - s.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        ctx = (a @ vr).transpose(1, 0, 2).reshape(S, HIDDEN)
        h = h + ctx @ w[p + "attn_output.weight"].T
        x = np_rms(h, w[p + "ffn_norm.weight"])
        g = x @ w[p + "ffn_gate.weight"].T
        u = x @ w[p + "ffn_up.weight"].T
        h = h + (g / (1.0 + np.exp(-g)) * u) @ w[p + "ffn_down.weight"].T
    return h, ks, vs


# ---------- MIL authoring ----------------------------------------------------

def author_bucket(S, w, outdir):
    f16 = lambda a: a.astype(np.float16)
    half = HEAD_DIM // 2
    cos, sin = rope_tables(S)
    mask = np.triu(np.full((S, S), -1e4, dtype=np.float32), 1)
    # [1, S, half, 1] to broadcast over heads and the pair axis.
    cosb = f16(cos)[None, :, :, None]
    sinb = f16(sin)[None, :, :, None]
    maskb = f16(mask)[None]
    scale = np.float16(1.0 / np.sqrt(HEAD_DIM))
    rep = HEADS // KV_HEADS

    @mb.program(input_specs=[mb.TensorSpec(shape=(S, HIDDEN), dtype=types.fp32)])
    def prog(h):
        x = mb.cast(x=h, dtype="fp16")

        def rms(t, g):
            sq = mb.mul(x=t, y=t)
            m = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
            r = mb.rsqrt(x=mb.add(x=m, y=np.float16(EPS)))
            return mb.mul(x=mb.mul(x=t, y=r), y=f16(g))

        def rope(t, nh):
            # Interleaved pairs: [nh,S,D] -> [nh,S,half,2]; freq i rotates
            # (x[2i], x[2i+1]) — mirrors rayzor_tensor_rope exactly.
            p = mb.reshape(x=t, shape=(nh, S, half, 2))
            e = mb.slice_by_index(
                x=p, begin=[0, 0, 0, 0], end=[0, 0, 0, 1],
                end_mask=[True, True, True, False],
            )
            o = mb.slice_by_index(
                x=p, begin=[0, 0, 0, 1], end=[0, 0, 0, 2],
                end_mask=[True, True, True, True],
            )
            re = mb.sub(x=mb.mul(x=e, y=cosb), y=mb.mul(x=o, y=sinb))
            im = mb.add(x=mb.mul(x=e, y=sinb), y=mb.mul(x=o, y=cosb))
            pr = mb.concat(values=[re, im], axis=-1)
            return mb.reshape(x=pr, shape=(nh, S, HEAD_DIM))

        outs = []
        for i in range(LAYERS):
            p = f"blk.{i}."
            xn = rms(x, w[p + "attn_norm.weight"])
            q = mb.linear(x=xn, weight=f16(w[p + "attn_q.weight"]))
            k = mb.linear(x=xn, weight=f16(w[p + "attn_k.weight"]))
            v = mb.linear(x=xn, weight=f16(w[p + "attn_v.weight"]))
            q = mb.transpose(x=mb.reshape(x=q, shape=(S, HEADS, HEAD_DIM)), perm=(1, 0, 2))
            k = mb.transpose(x=mb.reshape(x=k, shape=(S, KV_HEADS, HEAD_DIM)), perm=(1, 0, 2))
            v = mb.transpose(x=mb.reshape(x=v, shape=(S, KV_HEADS, HEAD_DIM)), perm=(1, 0, 2))
            q = rope(q, HEADS)
            k = rope(k, KV_HEADS)
            # KV outputs in cache layout [S, kv_heads, head_dim].
            outs.append(mb.transpose(x=k, perm=(1, 0, 2), name=f"k{i}"))
            outs.append(mb.transpose(x=v, perm=(1, 0, 2), name=f"v{i}"))
            kt = mb.reshape(
                x=mb.tile(x=mb.reshape(x=k, shape=(KV_HEADS, 1, S, HEAD_DIM)), reps=(1, rep, 1, 1)),
                shape=(HEADS, S, HEAD_DIM),
            )
            vt = mb.reshape(
                x=mb.tile(x=mb.reshape(x=v, shape=(KV_HEADS, 1, S, HEAD_DIM)), reps=(1, rep, 1, 1)),
                shape=(HEADS, S, HEAD_DIM),
            )
            s = mb.mul(x=mb.matmul(x=q, y=mb.transpose(x=kt, perm=(0, 2, 1))), y=scale)
            s = mb.add(x=s, y=maskb)
            a = mb.softmax(x=s, axis=-1)
            ctx = mb.matmul(x=a, y=vt)
            ctx = mb.reshape(x=mb.transpose(x=ctx, perm=(1, 0, 2)), shape=(S, HIDDEN))
            o = mb.linear(x=ctx, weight=f16(w[p + "attn_output.weight"]))
            x = mb.add(x=x, y=o)
            xn = rms(x, w[p + "ffn_norm.weight"])
            g = mb.linear(x=xn, weight=f16(w[p + "ffn_gate.weight"]))
            u = mb.linear(x=xn, weight=f16(w[p + "ffn_up.weight"]))
            ff = mb.linear(x=mb.mul(x=mb.silu(x=g), y=u), weight=f16(w[p + "ffn_down.weight"]))
            x = mb.add(x=x, y=ff)
        out = mb.cast(x=x, dtype="fp32", name="out")
        kv32 = [mb.cast(x=t, dtype="fp32", name=t.name + "f") for t in outs]
        return tuple([out] + kv32)

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Validate hidden + a sample of KV outputs against the numpy reference.
    rng = np.random.default_rng(9)
    h0 = ((rng.random((S, HIDDEN), dtype=np.float32) - 0.5) * 0.1)
    ref_h, ref_ks, ref_vs = np_forward(h0.copy(), w, rope_tables(S)[0], rope_tables(S)[1],
                                       np.triu(np.full((S, S), -1e4, dtype=np.float32), 1))
    got = m.predict({"h": h0})

    def cosv(a, b):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    ch = cosv(ref_h, got["out"])
    ck0 = cosv(ref_ks[0], got["k0f"])
    ckl = cosv(ref_ks[LAYERS - 1], got[f"k{LAYERS - 1}f"])
    cvl = cosv(ref_vs[LAYERS - 1], got[f"v{LAYERS - 1}f"])
    print(f"S={S}  cos: hidden={ch:.6f} k0={ck0:.6f} k_last={ckl:.6f} v_last={cvl:.6f}")
    assert min(ch, ck0, ckl, cvl) > 0.99, "prefill graph diverges from reference"

    path = f"{outdir}/{STEM}.prefill_s{S}.mlpackage"
    m.save(path)
    print("saved", path)


def main():
    global HIDDEN, HEADS, KV_HEADS, HEAD_DIM, FFN, LAYERS, EPS, ROPE_BASE, STEM
    gguf_path = sys.argv[1]
    outdir = sys.argv[2]
    buckets = [int(b) for b in sys.argv[3:]] or [128, 512]
    w, d = load_model(gguf_path)
    HIDDEN, LAYERS, HEADS, KV_HEADS, FFN = (
        d["hidden"], d["layers"], d["heads"], d["kv_heads"], d["ffn"],
    )
    HEAD_DIM = HIDDEN // HEADS
    EPS, ROPE_BASE = d["eps"], d["rope_base"]
    STEM = gguf_path.rsplit("/", 1)[-1].removesuffix(".gguf")
    print(f"model {STEM}: hidden={HIDDEN} layers={LAYERS} heads={HEADS}/{KV_HEADS} ffn={FFN} rope_base={ROPE_BASE}")
    for S in buckets:
        author_bucket(S, w, outdir)


if __name__ == "__main__":
    main()
