package nue.transformer;

import rayzor.ds.Tensor;

/**
 * Rotary position embedding — precomputes the cos/sin tables for a given
 * head dimension and maximum context length, then applies them on demand.
 *
 * RoPE is parameter-free (the cos/sin tables are derived, not learned),
 * so this is NOT a `Module` — it doesn't implement `parameters()`. Use it
 * as a helper held inside a `GQAttention` or sibling block.
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
 */
class RoPE {
    public var cos:Tensor;
    public var sin:Tensor;
    public var headDim:Int;
    public var maxSeqLen:Int;
    public var base:Float;

    public function new(headDim:Int, maxSeqLen:Int, base:Float = 10000.0) {
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.base = base;
        this.cos = Tensor.ropeCosTable(headDim, maxSeqLen, base);
        this.sin = Tensor.ropeSinTable(headDim, maxSeqLen, base);
    }

    /**
     * Apply rotary embedding to `x` of shape `[seq_len, num_heads, head_dim]`
     * (or `[seq_len, head_dim]`). `positionOffset` adds to the per-row
     * position — used for incremental decode.
     */
    public function apply(x:Tensor, positionOffset:Int = 0):Tensor {
        return x.rope(cos, sin, positionOffset);
    }
}
