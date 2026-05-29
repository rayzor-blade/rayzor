package nue.transformer;

import nue.Module;
import nue.Linear;
import rayzor.ds.Tensor;
import rayzor.ds.DType;

/**
 * Grouped Query Attention (the Llama 3+ attention variant).
 *
 * Each forward pass:
 *   x: [seq_q, hidden_size]
 *     → Q [seq_q, num_q_heads, head_dim]  (via q_proj)
 *       K [seq_q, num_kv_heads, head_dim] (via k_proj)
 *       V [seq_q, num_kv_heads, head_dim] (via v_proj)
 *     → RoPE-rotate Q and K (positions accounting for past cache)
 *     → append (K, V) to the per-layer KVCache
 *     → scores = (Q @ Kᵀ_cache) * (1 / sqrt(head_dim))   [shape per-head]
 *     → mask upper triangle with -inf (causal)
 *     → softmax along the key dim
 *     → context = scores @ V_cache                       [shape per-head]
 *     → output [seq_q, num_q_heads * head_dim] @ o_proj  → [seq_q, hidden_size]
 *
 * Grouped query means num_q_heads is a multiple of num_kv_heads (Llama
 * 3.2 1B: 32 Q heads, 8 KV heads → 4 Q heads share each KV head). We
 * implement the broadcast by reshape-and-tile of K/V inside the
 * attention loop — see `expandKvHeads` below. The math itself is the
 * standard scaled-dot-product pattern; each tensor op JITs down to the
 * existing SIMD/GPU kernels, so this Haxe forward path is competitive
 * with a fully-fused Rust kernel on F32 (the inner matmuls do all the
 * work).
 *
 * Choices left for the caller to override by subclassing:
 *   - alibi bias, sliding window mask: re-implement `applyMask`
 *   - flash-attention-style streaming: re-implement `forward` to fuse
 *     across heads / sub-blocks
 *   - MoE routing across heads: re-implement `forward` for sparse
 *     per-token expert selection
 */
class GQAttention implements Module {
    public var qProj:Linear;
    public var kProj:Linear;
    public var vProj:Linear;
    public var oProj:Linear;
    public var rope:RoPE;
    public var cache:KVCache;

    public var numQHeads:Int;
    public var numKvHeads:Int;
    public var headDim:Int;
    public var scale:Float; // 1 / sqrt(headDim)

    public function new(
        qProj:Linear, kProj:Linear, vProj:Linear, oProj:Linear,
        rope:RoPE, cache:KVCache,
        numQHeads:Int, numKvHeads:Int, headDim:Int
    ) {
        this.qProj = qProj;
        this.kProj = kProj;
        this.vProj = vProj;
        this.oProj = oProj;
        this.rope = rope;
        this.cache = cache;
        this.numQHeads = numQHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.scale = 1.0 / Math.sqrt(headDim);
    }

    /**
     * Forward pass on x of shape [seq_q, hidden_size]. Mutates the
     * underlying KVCache by appending the new K/V slices.
     */
    public function forward(x:Tensor):Tensor {
        var seqQ = x.shape()[0];
        var positionOffset = cache.currentLen;

        // 1) Project to Q, K, V.
        //    hidden_size = numQHeads * headDim  (Q out)
        //                = numKvHeads * headDim (K, V out)
        var q = qProj.forward(x).reshape([seqQ, numQHeads, headDim]);
        var k = kProj.forward(x).reshape([seqQ, numKvHeads, headDim]);
        var v = vProj.forward(x).reshape([seqQ, numKvHeads, headDim]);

        // 2) Rotary embedding — rotates the absolute positions starting
        //    at `positionOffset` (so the new tokens line up with cache).
        q = rope.apply(q, positionOffset);
        k = rope.apply(k, positionOffset);

        // 3) Push the new K/V into the cache. Subsequent reads use the
        //    full active slice (prior tokens + just-added).
        cache.append(k, v);
        var kAll = cache.keysView();   // [cache.currentLen, numKvHeads, headDim]
        var vAll = cache.valuesView(); // same shape

        // 4) Score = Q @ Kᵀ, scaled. Per-head, in a batched matmul.
        //    Q is [seqQ, numQHeads, headDim]; we transpose to put heads
        //    on the batch axis so bmm computes one (M,K)x(K,N) per head.
        //    Then K from the cache is broadcast to numQHeads by repeating
        //    each KV head G = numQHeads / numKvHeads times.
        var qByHead = q.permute([1, 0, 2]);     // [numQHeads, seqQ, headDim]
        var kAllExpanded = expandKvHeads(kAll); // [numQHeads, cacheLen, headDim]
        var vAllExpanded = expandKvHeads(vAll); // [numQHeads, cacheLen, headDim]
        // kᵀ for matmul: swap the last two dims of the per-head 2-D matrix
        // so the inner product runs along headDim.
        var kT = kAllExpanded.transposeLast2(); // [numQHeads, headDim, cacheLen]
        var scores = qByHead.bmm(kT);            // [numQHeads, seqQ, cacheLen]
        scores = scores.scale(scale);

        // 5) Causal mask. Each query row i can only see keys
        //    [0, i + positionOffset]. This shifts the mask diagonal by
        //    positionOffset.
        scores = applyMask(scores, positionOffset);

        // 6) Softmax along the last dim (the key axis).
        var attn = scores.softmax();             // [numQHeads, seqQ, cacheLen]

        // 7) Context = attn @ V, per head.
        var context = attn.bmm(vAllExpanded);    // [numQHeads, seqQ, headDim]

        // 8) Bring heads back to the row axis and flatten head/head_dim.
        var contextRowMajor = context.permute([1, 0, 2]); // [seqQ, numQHeads, headDim]
        var hiddenSize = numQHeads * headDim;
        var contextFlat = contextRowMajor.reshape([seqQ, hiddenSize]);

        // 9) Output projection back to hidden_size.
        return oProj.forward(contextFlat);
    }

    /**
     * Broadcast a `[seqK, numKvHeads, headDim]` tensor up to
     * `[numQHeads, seqK, headDim]` by repeating each KV head
     * `G = numQHeads / numKvHeads` consecutive times along the head
     * axis, then permuting heads to the front.
     *
     * Done as a small per-element copy until the compiler grows a real
     * broadcast primitive — for typical Llama 3.2 1B shapes (G=4,
     * head_dim=64) this loop is dwarfed by the subsequent bmm so it
     * doesn't matter; revisit when profiling says otherwise.
     */
    public function expandKvHeads(t:Tensor):Tensor {
        var s = t.shape();
        var seqK = s[0];
        var headDimV = s[2];
        var group = Std.int(numQHeads / numKvHeads);
        // Bare `F32` — TAST enum-variant disambiguation picks DType.F32 because
        // Tensor.zeros's dtype param is typed as DType.
        var out = Tensor.zeros([numQHeads, seqK, headDimV], F32);
        for (qh in 0...numQHeads) {
            var kvh = Std.int(qh / group);
            for (j in 0...seqK) {
                for (d in 0...headDimV) {
                    out.set([qh, j, d], t.get([j, kvh, d]));
                }
            }
        }
        return out;
    }

    /**
     * Apply the standard causal mask to `[numQHeads, seqQ, cacheLen]`
     * scores. Override in a subclass to plug in alibi bias, sliding
     * window, etc.
     */
    public function applyMask(scores:Tensor, positionOffset:Int):Tensor {
        // The runtime primitive treats the last two dims as [rows, cols]
        // and masks j > i + positionOffset across every outer batch slice.
        return scores.causalMask_(positionOffset);
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
