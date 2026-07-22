# Llama PREFILL-on-ANE probe (the 2-block experiment) — answers, for ~a day
# instead of a week: can CoreML/ANE run real Llama-3.2-1B blocks (RMSNorm +
# RoPE + GQA + causal mask + SwiGLU) authored straight from the Q4_K_M GGUF,
# correctly, and at what rate — extrapolated x8 to a 16-block prefill ttft.
#
# Decode stays on CPU by design (bandwidth-bound GEMV); prefill is the
# encode-shaped compute-bound phase, and on unified memory the
# ANE-prefill -> CPU-decode split hands the KV cache over for free.
#
# Scope notes:
# - Weights: Q4_K dequantized (gguf.quants) -> fp16 baked. ~121 MB/block.
# - RoPE: NEOX half-rotation with baked cos/sin tables per bucket, plain
#   freq_base=500000 (this gguf carries no rope-scaling keys). The numpy
#   reference uses the SAME formula — the probe validates authoring + speed;
#   exact convention-matching against the Haxe path happens at integration
#   (bugs_rope_interleaved_for_gguf is the cautionary tale).
# - Causal mask baked as a [S, S] additive constant (0 / -1e4, fp16-safe).
# - GQA: kv heads tiled 4x so kv head i serves q heads 4i..4i+3.
#
# Usage:
#   mlvenv/bin/python llama_prefill_probe.py <llama.gguf> <outdir> [S=128] [blocks=2]

import os
import sys
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from gguf import GGUFReader
from gguf import quants as gq

HIDDEN = 2048
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
FFN = 8192
EPS = 1e-5
ROPE_BASE = 500000.0


def load_blocks(gguf_path, n_blocks):
    r = GGUFReader(gguf_path)
    w = {}
    want = []
    for i in range(n_blocks):
        p = f"blk.{i}."
        want += [
            p + n
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
            )
        ]
    for t in r.tensors:
        if t.name not in want:
            continue
        data = gq.dequantize(t.data, t.tensor_type)
        w[t.name] = np.asarray(data, dtype=np.float32).reshape(
            tuple(int(d) for d in reversed(t.shape))
        )
    missing = [n for n in want if n not in w]
    assert not missing, f"missing tensors: {missing[:4]}"
    return w


def rope_tables(S):
    half = HEAD_DIM // 2
    inv = 1.0 / (ROPE_BASE ** (np.arange(half, dtype=np.float64) / half))
    ang = np.outer(np.arange(S, dtype=np.float64), inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


# ---------- numpy reference --------------------------------------------------

def np_rms(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g


def np_rope(x, cos, sin):
    # x: [H, S, D]; NEOX half-rotation.
    half = HEAD_DIM // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def np_blocks(h, w, n_blocks, cos, sin, mask):
    S = h.shape[0]
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for i in range(n_blocks):
        p = f"blk.{i}."
        x = np_rms(h, w[p + "attn_norm.weight"])
        q = (x @ w[p + "attn_q.weight"].T).reshape(S, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = (x @ w[p + "attn_k.weight"].T).reshape(S, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = (x @ w[p + "attn_v.weight"].T).reshape(S, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
        q, k = np_rope(q, cos, sin), np_rope(k, cos, sin)
        k = np.repeat(k, HEADS // KV_HEADS, 0)
        v = np.repeat(v, HEADS // KV_HEADS, 0)
        s = q @ k.transpose(0, 2, 1) * scale + mask[None]
        a = np.exp(s - s.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        ctx = (a @ v).transpose(1, 0, 2).reshape(S, HIDDEN)
        h = h + ctx @ w[p + "attn_output.weight"].T
        x = np_rms(h, w[p + "ffn_norm.weight"])
        g = x @ w[p + "ffn_gate.weight"].T
        u = x @ w[p + "ffn_up.weight"].T
        h = h + (g / (1.0 + np.exp(-g)) * u) @ w[p + "ffn_down.weight"].T
    return h


# ---------- MIL authoring ----------------------------------------------------

def author(S, w, n_blocks, cos, sin, mask):
    f16 = lambda a: a.astype(np.float16)
    half = HEAD_DIM // 2
    cosb = f16(cos)[None]  # [1, S, half] broadcast over heads
    sinb = f16(sin)[None]
    maskb = f16(mask)[None]  # [1, S, S]
    scale = np.float16(1.0 / np.sqrt(HEAD_DIM))

    STRIP = set(os.environ.get("STRIP", "").split(","))

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(S, HIDDEN), dtype=types.fp32)],
    )
    def prog(h):
        x = mb.cast(x=h, dtype="fp16")

        def rms(t, g, nm):
            if "rms" in STRIP:
                return t
            sq = mb.mul(x=t, y=t)
            m = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
            r = mb.rsqrt(x=mb.add(x=m, y=np.float16(EPS)))
            return mb.mul(x=mb.mul(x=t, y=r), y=f16(g), name=nm)

        def rope(t, nm):
            if "rope" in STRIP:
                return t
            t1 = mb.slice_by_index(x=t, begin=[0, 0, 0], end=[0, 0, half], end_mask=[True, True, False])
            t2 = mb.slice_by_index(x=t, begin=[0, 0, half], end=[0, 0, HEAD_DIM], end_mask=[True, True, True])
            r1 = mb.sub(x=mb.mul(x=t1, y=cosb), y=mb.mul(x=t2, y=sinb))
            r2 = mb.add(x=mb.mul(x=t1, y=sinb), y=mb.mul(x=t2, y=cosb))
            return mb.concat(values=[r1, r2], axis=-1, name=nm)

        for i in range(n_blocks):
            p = f"blk.{i}."
            n = f"b{i}_"
            xn = rms(x, w[p + "attn_norm.weight"], n + "an")
            q = mb.linear(x=xn, weight=f16(w[p + "attn_q.weight"]))
            k = mb.linear(x=xn, weight=f16(w[p + "attn_k.weight"]))
            v = mb.linear(x=xn, weight=f16(w[p + "attn_v.weight"]))
            q = mb.transpose(x=mb.reshape(x=q, shape=(S, HEADS, HEAD_DIM)), perm=(1, 0, 2))
            k = mb.transpose(x=mb.reshape(x=k, shape=(S, KV_HEADS, HEAD_DIM)), perm=(1, 0, 2))
            v = mb.transpose(x=mb.reshape(x=v, shape=(S, KV_HEADS, HEAD_DIM)), perm=(1, 0, 2))
            q = rope(q, n + "qr")
            k = rope(k, n + "kr")
            # GQA tile: [8,S,64] -> [8,1,S,64] -> [8,4,S,64] -> [32,S,64]
            rep = HEADS // KV_HEADS
            if "tile" in STRIP:
                k = mb.concat(values=[k, k, k, k], axis=0)
                v = mb.concat(values=[v, v, v, v], axis=0)
            else:
                k = mb.reshape(x=mb.tile(x=mb.reshape(x=k, shape=(KV_HEADS, 1, S, HEAD_DIM)), reps=(1, rep, 1, 1)), shape=(HEADS, S, HEAD_DIM))
                v = mb.reshape(x=mb.tile(x=mb.reshape(x=v, shape=(KV_HEADS, 1, S, HEAD_DIM)), reps=(1, rep, 1, 1)), shape=(HEADS, S, HEAD_DIM))
            # Rank-4 scaled_dot_product_attention so CoreML's compiler can
            # pattern-match its ANE attention kernel — the hand-rolled rank-3
            # matmul+softmax chain is rejected by ANECCompile.
            s = mb.mul(x=mb.matmul(x=q, y=mb.transpose(x=k, perm=(0, 2, 1))), y=scale)
            s = mb.add(x=s, y=maskb)
            a = mb.softmax(x=s, axis=-1)
            ctx = mb.matmul(x=a, y=v)
            ctx = mb.reshape(x=mb.transpose(x=ctx, perm=(1, 0, 2)), shape=(S, HIDDEN))
            o = mb.linear(x=ctx, weight=f16(w[p + "attn_output.weight"]))
            x = mb.add(x=x, y=o)
            xn = rms(x, w[p + "ffn_norm.weight"], n + "fn")
            g = mb.linear(x=xn, weight=f16(w[p + "ffn_gate.weight"]))
            u = mb.linear(x=xn, weight=f16(w[p + "ffn_up.weight"]))
            act = mb.gelu(x=g, mode="EXACT") if "silu" in STRIP else mb.silu(x=g)
            ff = mb.linear(x=mb.mul(x=act, y=u), weight=f16(w[p + "ffn_down.weight"]))
            x = mb.add(x=x, y=ff)
        return mb.cast(x=x, dtype="fp32", name="out")

    return ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.macOS15,
    )


def main():
    gguf_path = sys.argv[1]
    outdir = sys.argv[2]
    S = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    n_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    w = load_blocks(gguf_path, n_blocks)
    cos, sin = rope_tables(S)
    mask = np.triu(np.full((S, S), -1e4, dtype=np.float32), 1)

    m = author(S, w, n_blocks, cos, sin, mask)

    rng = np.random.default_rng(3)
    h0 = ((rng.random((S, HIDDEN), dtype=np.float32) - 0.5) * 0.1)
    ref = np_blocks(h0.copy(), w, n_blocks, cos, sin, mask)
    got = list(m.predict({"h": h0}).values())[0]
    cosim = float(ref.ravel() @ got.ravel() / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    print(f"S={S} blocks={n_blocks}  cosine(mlprogram, numpy-ref) = {cosim:.6f}")
    if not os.environ.get("STRIP"):
        assert cosim > 0.995, f"authored blocks diverge (cos={cosim})"

    pkg = f"{outdir}/llama_prefill_probe_s{S}_b{n_blocks}.mlpackage"
    m.save(pkg)
    print("saved", pkg)

    for cu, label in [
        (ct.ComputeUnit.CPU_ONLY, "cpu-only"),
        (ct.ComputeUnit.CPU_AND_NE, "CPU+NE (ANE)"),
    ]:
        mm = ct.models.MLModel(pkg, compute_units=cu)
        mm.predict({"h": h0})
        for _ in range(8):
            mm.predict({"h": h0})
        t0 = time.perf_counter()
        N = 30
        for _ in range(N):
            mm.predict({"h": h0})
        per = (time.perf_counter() - t0) / N
        full = per * (16 / n_blocks)
        print(
            f"{label:14s} {per*1e3:8.2f} ms/{n_blocks}-blocks   -> 16-block prefill est {full*1e3:7.1f} ms  ({S/full:6.0f} tok/s prefill)"
        )


if __name__ == "__main__":
    main()
