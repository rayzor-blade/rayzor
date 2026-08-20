package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.Mem;
import rayzor.Usize;

/**
 * Rotary position embedding — precomputes the cos/sin tables for a given
 * head dimension and maximum context length, then applies them on demand.
 *
 * RoPE is parameter-free (the cos/sin tables are derived, not learned),
 * so this is NOT a `Module` — it doesn't implement `parameters()`. Use
 * it as a helper held inside a `GQAttention` or sibling block.
 *
 * Example:
 * ```haxe
 * var rope = new RoPE(64, 2048, 10000.0);   // head_dim=64, ctx=2048
 * var q_rot = rope.apply(q, /* positionOffset */ 0);
 * var k_rot = rope.apply(k, /* positionOffset */ 0);
 * ```
 *
 * During incremental decode, pass the absolute position of the new token
 * as `positionOffset` so the rotation matches the previously-cached
 * positions.
 *
 * The rotation itself is written here, in Haxe. It is arithmetic — a 2x2
 * rotation per lane pair against a derived table — not a platform capability,
 * so it belongs on this side of the FFI boundary along with the rest of the
 * kernels. Keeping it in the tensor plugin also fixed the rotated width at
 * `headDim / 2`, which made partial rotary inexpressible; here the width is
 * an argument.
 *
 * **GPU dispatch.** CPU-only by design. For on-device inference, call
 * `rayzor.gpu.GPUCompute.rope(xGpu, cosGpu, sinGpu, ...)` directly with
 * the LUTs pre-uploaded via `gpu.createBuffer(rope.cos / .sin)`. The
 * F16-stored LUT variants from `Tensor.ropeCosTableF16` halve VRAM use.
 * Module-level wrapper deferred pending the JIT quirks documented in
 * `bugs_known.md`.
 */
class RoPE {
    public var cos:Tensor;
    public var sin:Tensor;
    public var headDim:Int;
    public var maxSeqLen:Int;
    public var base:Float;

    /** RoPE pairing convention. `false` = interleaved/NORM (Llama, Mistral;
     *  GGUF permutes Q/K). `true` = half-split/NEOX (Qwen2, GPT-NeoX, Falcon;
     *  GGUF leaves Q/K unpermuted). Set by the arch builder after construction;
     *  applying the wrong one gives confident-but-wrong logits (degenerate
     *  attention), the classic symptom of a RoPE-convention mismatch. */
    public var neox:Bool;

    /** How many of the head's lanes rotate. Defaults to all of them; a model
        with partial rotary (Qwen3.5) rotates a prefix and passes the rest
        through untouched. Must be even, and no larger than `headDim`. */
    public var rotaryDim:Int;

    public function new(headDim:Int, maxSeqLen:Int, base:Float = 10000.0, rotaryDim:Int = 0) {
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.base = base;
        this.neox = false;
        this.rotaryDim = (rotaryDim > 0 && rotaryDim <= headDim) ? rotaryDim : headDim;
        // The table is derived, not learned, and the plugin already builds
        // exactly the shape partial rotary needs when asked for `rotaryDim`
        // rather than `headDim`: `[maxSeqLen, rotaryDim/2]`.
        this.cos = Tensor.ropeCosTable(this.rotaryDim, maxSeqLen, base);
        this.sin = Tensor.ropeSinTable(this.rotaryDim, maxSeqLen, base);
    }

    /** `[maxSeqLen, rotaryDim/2]` of `cos(p * base^(-2i/rotaryDim))`, or the
        sine of the same angle. The frequency is indexed by lane pair, so the
        two tables differ only in which trig function closes them. */
    static function buildTable(rotDim:Int, maxSeqLen:Int, base:Float, wantSin:Bool):Tensor {
        var half = rotDim >> 1;
        var t = Tensor.zeros([maxSeqLen, half], DType.F32);
        var dst:Usize = t.data().raw();
        var dimF = rotDim * 1.0;
        for (i in 0...half) {
            var theta = 1.0 / Math.pow(base, (2 * i) * 1.0 / dimF);
            for (p in 0...maxSeqLen) {
                var a = p * theta;
                Mem.storeF32(dst + Usize.fromInt((p * half + i) * 4),
                    wantSin ? Math.sin(a) : Math.cos(a));
            }
        }
        return t;
    }

    /**
     * Apply rotary embedding to `x` of shape `[seq_len, num_heads, head_dim]`
     * (or `[seq_len, head_dim]`). `positionOffset` adds to the per-row
     * position — used for incremental decode.
     */
    public function apply(x:Tensor, positionOffset:Int = 0):Tensor {
        var shape = x.shape();
        var nd = shape.length;
        if (nd < 2) return x.clone();
        var hd = shape[nd - 1];
        var heads = nd >= 3 ? shape[nd - 2] : 1;
        var seq = 1;
        if (nd >= 3) {
            for (i in 0...nd - 2) seq *= shape[i];
        } else {
            seq = shape[0];
        }
        var rot = rotaryDim <= hd ? rotaryDim : hd;
        var half = rot >> 1;
        if (half <= 0) return x.clone();

        var out = Tensor.zeros(shape, DType.F32);
        var src:Usize = x.data().raw();
        var dst:Usize = out.data().raw();
        var cosB:Usize = cos.data().raw();
        var sinB:Usize = sin.data().raw();
        var perRow = heads * hd;

        for (s in 0...seq) {
            var pos = s + positionOffset;
            // Past the table, the row passes through. The alternative — clamping
            // to the last row — would rotate every further token by the same
            // angle, which reads as a plausible output and is wrong.
            var inTable = pos >= 0 && pos < maxSeqLen;
            for (h in 0...heads) {
                var base = (s * perRow + h * hd) * 4;
                if (!inTable) {
                    for (i in 0...hd) {
                        var o = base + i * 4;
                        Mem.storeF32(dst + Usize.fromInt(o), Mem.loadF32(src + Usize.fromInt(o)));
                    }
                    continue;
                }
                for (i in 0...half) {
                    // f32 throughout, not Haxe's f64 `Float`. The rotation has
                    // to be the SAME function as every other implementation of
                    // it, and a double-precision intermediate rounded once at
                    // the end is a different one — accurate enough to look
                    // right and different enough to move a sampled token.
                    var cv:Single = Mem.loadF32(cosB + Usize.fromInt((pos * half + i) * 4));
                    var sv:Single = Mem.loadF32(sinB + Usize.fromInt((pos * half + i) * 4));
                    // NEOX pairs lanes a half apart, NORM pairs them adjacent.
                    // The table is the same either way; only the pairing moves.
                    var lo = neox ? base + i * 4 : base + (2 * i) * 4;
                    var hi = neox ? base + (half + i) * 4 : base + (2 * i + 1) * 4;
                    var xl:Single = Mem.loadF32(src + Usize.fromInt(lo));
                    var xh:Single = Mem.loadF32(src + Usize.fromInt(hi));
                    var rl:Single = xl * cv - xh * sv;
                    var rh:Single = xl * sv + xh * cv;
                    Mem.storeF32(dst + Usize.fromInt(lo), rl);
                    Mem.storeF32(dst + Usize.fromInt(hi), rh);
                }
                // Lanes past the rotated width are carried over unchanged.
                for (i in rot...hd) {
                    var o = base + i * 4;
                    Mem.storeF32(dst + Usize.fromInt(o), Mem.loadF32(src + Usize.fromInt(o)));
                }
            }
        }
        return out;
    }
}
