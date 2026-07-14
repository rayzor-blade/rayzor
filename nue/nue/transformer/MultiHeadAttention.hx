package nue.transformer;

import nue.Module;
import nue.Linear;
import rayzor.ds.Tensor;

/**
 * Standard multi-head attention — no grouped-query, no RoPE, no
 * causal mask, no KV cache. The encoder shape used by BERT / RoBERTa
 * / DeBERTa / vanilla Transformer.
 *
 * For a decoder shape (causal mask + KV cache + RoPE + grouped query),
 * see `GQAttention`. The two attention modules deliberately live
 * apart instead of `GQAttention` growing flags — the encoder code
 * path is half the size when it doesn't have to skip the causal /
 * cache logic, and a subclass that "turns off" half its parent is
 * worse documentation than two siblings.
 *
 * `forward` attends bidirectionally over the whole sequence.
 * `forwardMasked` adds a caller-supplied additive bias to the scores
 * before softmax — a `[seq_key]` vector, `0` for real keys and a large
 * negative for padded keys, broadcast over heads and query positions —
 * so batched/padded inputs ignore padding. The bias is an argument, not
 * instance state, so concurrent encodes on a shared model don't race.
 */
class MultiHeadAttention implements Attention {
    public var qProj:Linear;
    public var kProj:Linear;
    public var vProj:Linear;
    public var oProj:Linear;
    public var numHeads:Int;
    public var headDim:Int;
    public var scale:Float;

    public function new(
        qProj:Linear, kProj:Linear, vProj:Linear, oProj:Linear,
        numHeads:Int, headDim:Int
    ) {
        this.qProj = qProj;
        this.kProj = kProj;
        this.vProj = vProj;
        this.oProj = oProj;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.scale = 1.0 / Math.sqrt(headDim);
    }

    public function forward(x:Tensor):Tensor {
        var seq = x.shape()[0];

        // Three independent consumers of x → clone twice; last use moves
        // the original.
        var q = qProj.forward(x.clone()).reshape([seq, numHeads, headDim]).permute([1, 0, 2]);
        var k = kProj.forward(x.clone()).reshape([seq, numHeads, headDim]).permute([1, 0, 2]);
        var v = vProj.forward(x).reshape([seq, numHeads, headDim]).permute([1, 0, 2]);

        // scores: [heads, seq, seq]
        var scores = q.bmm(k.transposeLast2()).scale(scale);
        // No mask — bidirectional attention.
        var attn = scores.softmax();
        var context = attn.bmm(v).permute([1, 0, 2]);

        var flat = context.reshape([seq, numHeads * headDim]);
        return oProj.forward(flat);
    }

    /**
     * Attention with an additive padding mask. `attnBias` is a contiguous
     * `[seq]` F32 vector added to every `[head, query]` row of the
     * `[heads, seq, seq]` scores (addInto broadcasts the trailing axis), so
     * padded keys carry a large-negative bias and softmax zeroes them. Mirrors
     * `forward` exactly but for the one `addInto` before softmax.
     */
    public function forwardMasked(x:Tensor, attnBias:Tensor):Tensor {
        var seq = x.shape()[0];

        var q = qProj.forward(x.clone()).reshape([seq, numHeads, headDim]).permute([1, 0, 2]);
        var k = kProj.forward(x.clone()).reshape([seq, numHeads, headDim]).permute([1, 0, 2]);
        var v = vProj.forward(x).reshape([seq, numHeads, headDim]).permute([1, 0, 2]);

        var scores = q.bmm(k.transposeLast2()).scale(scale);
        scores.addInto(attnBias); // + additive key mask, broadcast over [heads, query]
        var attn = scores.softmax();
        var context = attn.bmm(v).permute([1, 0, 2]);

        var flat = context.reshape([seq, numHeads * headDim]);
        return oProj.forward(flat);
    }

    public function parameters():Array<NamedTensor> {
        var ps = [];
        for (p in qProj.parameters()) ps.push({ name: "attn_q." + p.name, tensor: p.tensor });
        for (p in kProj.parameters()) ps.push({ name: "attn_k." + p.name, tensor: p.tensor });
        for (p in vProj.parameters()) ps.push({ name: "attn_v." + p.name, tensor: p.tensor });
        for (p in oProj.parameters()) ps.push({ name: "attn_output." + p.name, tensor: p.tensor });
        return ps;
    }
}
