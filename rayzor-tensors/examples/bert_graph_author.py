# Author the all-MiniLM-L6-v2 ENCODER STACK (6 post-norm BERT layers) as a
# CoreML mlprogram per sequence bucket, with weights baked straight from the
# GGUF — the same artifact the Haxe path loads, so there is one source of
# truth.
#
# Split of responsibilities (BERT-phased-plan Phase 4): the Haxe side keeps
# tokenization + embeddings (token+pos+segment+LayerNorm, gather-based and
# cheap) and mean-pool+L2; the GRAPH runs the 6 transformer layers — the
# ~95%-of-FLOPs part with fixed [S, 384] shapes that CoreML wants.
#
#   inputs : h    [S, 384] f32  (post-embedding hidden states)
#            bias [S]      f32  (additive key mask: 0 real, -1e4 pad)
#   output : out  [S, 384] f32  (final hidden states, pre-pooling)
#
# I/O stays FLOAT32 (BNNSGraph io default — landmine from bnns_graph_bench);
# compute is FP16 (fine for embeddings; cosine is the metric).
#
# Usage:
#   mlvenv/bin/python bert_graph_author.py <model.gguf> <outdir> [buckets...]
# then compile each package:
#   xcrun coremlc compile <outdir>/bert_encoder_s{S}.mlpackage <outdir>
#
# The script validates each bucket against a numpy reference implementation
# built from the SAME weights on random inputs (cosine > 0.999) before saving,
# so layout/transpose mistakes fail here, not in the Rust probe.

import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from gguf import GGUFReader

HIDDEN = 384
HEADS = 12
HEAD_DIM = 32
LAYERS = 6
FFN = 1536
EPS = 1e-12


def load_weights(gguf_path):
    r = GGUFReader(gguf_path)
    w = {}
    for t in r.tensors:
        w[t.name] = np.array(t.data, dtype=np.float32).reshape(
            tuple(int(d) for d in reversed(t.shape))
        )
    return w


def layer_params(w, i):
    p = f"blk.{i}."
    def lin(base):
        return w[p + base + ".weight"], w[p + base + ".bias"]
    return {
        "q": lin("attn_q"),
        "k": lin("attn_k"),
        "v": lin("attn_v"),
        "o": lin("attn_output"),
        "attn_norm": (w[p + "attn_output_norm.weight"], w[p + "attn_output_norm.bias"]),
        "up": lin("ffn_up"),
        "down": lin("ffn_down"),
        "ffn_norm": (w[p + "layer_output_norm.weight"], w[p + "layer_output_norm.bias"]),
    }


# ---------- numpy reference (correctness oracle for the authored graph) ------

def np_layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * g + b


def np_gelu(x):
    from scipy.special import erf  # scipy ships with coremltools deps? fall back below
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def np_gelu_fallback(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def np_encoder(h, bias, params):
    try:
        gelu = np_gelu
        gelu(np.zeros(1, dtype=np.float32))
    except Exception:
        gelu = np_gelu_fallback
    S = h.shape[0]
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for lp in params:
        qw, qb = lp["q"]; kw, kb = lp["k"]; vw, vb = lp["v"]; ow, ob = lp["o"]
        q = (h @ qw.T + qb).reshape(S, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = (h @ kw.T + kb).reshape(S, HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = (h @ vw.T + vb).reshape(S, HEADS, HEAD_DIM).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) * scale + bias[None, None, :]
        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn = attn / attn.sum(-1, keepdims=True)
        ctx = (attn @ v).transpose(1, 0, 2).reshape(S, HIDDEN)
        h = np_layer_norm(h + (ctx @ ow.T + ob), *lp["attn_norm"])
        ff = gelu(h @ lp["up"][0].T + lp["up"][1]) @ lp["down"][0].T + lp["down"][1]
        h = np_layer_norm(h + ff, *lp["ffn_norm"])
    return h


# ---------- MIL authoring ----------------------------------------------------

def author_bucket(S, params, outdir):
    f16 = lambda a: a.astype(np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(S, HIDDEN), dtype=types.fp32),
            mb.TensorSpec(shape=(S,), dtype=types.fp32),
        ]
    )
    def prog(h, bias):
        x = mb.cast(x=h, dtype="fp16")
        bmask = mb.reshape(x=mb.cast(x=bias, dtype="fp16"), shape=(1, 1, S))
        scale = np.float16(1.0 / np.sqrt(HEAD_DIM))
        for li, lp in enumerate(params):
            n = f"l{li}_"
            qw, qb = lp["q"]; kw, kb = lp["k"]; vw, vb = lp["v"]; ow, ob = lp["o"]
            q = mb.linear(x=x, weight=f16(qw), bias=f16(qb), name=n + "q")
            k = mb.linear(x=x, weight=f16(kw), bias=f16(kb), name=n + "k")
            v = mb.linear(x=x, weight=f16(vw), bias=f16(vb), name=n + "v")
            q = mb.transpose(x=mb.reshape(x=q, shape=(S, HEADS, HEAD_DIM)), perm=(1, 0, 2))
            k = mb.transpose(x=mb.reshape(x=k, shape=(S, HEADS, HEAD_DIM)), perm=(1, 2, 0))
            v = mb.transpose(x=mb.reshape(x=v, shape=(S, HEADS, HEAD_DIM)), perm=(1, 0, 2))
            scores = mb.mul(x=mb.matmul(x=q, y=k), y=scale)
            scores = mb.add(x=scores, y=bmask)
            attn = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=attn, y=v)
            ctx = mb.reshape(x=mb.transpose(x=ctx, perm=(1, 0, 2)), shape=(S, HIDDEN))
            o = mb.linear(x=ctx, weight=f16(ow), bias=f16(ob), name=n + "o")
            x = mb.add(x=x, y=o)
            g, b = lp["attn_norm"]
            x = mb.layer_norm(x=x, axes=[-1], gamma=f16(g), beta=f16(b), epsilon=np.float16(1e-7))
            upw, upb = lp["up"]; dw, db = lp["down"]
            ff = mb.linear(x=x, weight=f16(upw), bias=f16(upb), name=n + "up")
            ff = mb.gelu(x=ff, mode="EXACT")
            ff = mb.linear(x=ff, weight=f16(dw), bias=f16(db), name=n + "down")
            x = mb.add(x=x, y=ff)
            g, b = lp["ffn_norm"]
            x = mb.layer_norm(x=x, axes=[-1], gamma=f16(g), beta=f16(b), epsilon=np.float16(1e-7))
        return mb.cast(x=x, dtype="fp32", name="out")

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Validate vs the numpy reference on random data before saving.
    rng = np.random.default_rng(11)
    h0 = (rng.random((S, HIDDEN), dtype=np.float32) - 0.5) * 0.2
    bias0 = np.zeros(S, dtype=np.float32)
    bias0[S - S // 8 :] = -1e4  # tail padding masked
    ref = np_encoder(h0.copy(), bias0, params)
    got = list(m.predict({"h": h0, "bias": bias0}).values())[0]
    cos = float(
        (ref.ravel() @ got.ravel())
        / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12)
    )
    print(f"S={S}  cosine(mlprogram, numpy-ref) = {cos:.6f}")
    assert cos > 0.999, f"authored graph diverges from reference (S={S}, cos={cos})"

    path = f"{outdir}/bert_encoder_s{S}.mlpackage"
    m.save(path)
    print(f"saved {path}")


def main():
    gguf_path = sys.argv[1]
    outdir = sys.argv[2]
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
    w = load_weights(gguf_path)
    params = [layer_params(w, i) for i in range(LAYERS)]
    for S in buckets:
        author_bucket(S, params, outdir)


if __name__ == "__main__":
    main()
