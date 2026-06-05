package nue.transformer;

import nue.Module;
import nue.Linear;
import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
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
        //
        //    Fast path: when all three projections are Q4_K_M (the typical
        //    Llama-3 deployment) dispatch through `QTensor.matmulFusedQKV`,
        //    which pre-quantises x to Q8_K exactly once and shares that
        //    view across all three weight matrices in a single
        //    `parallel_rows` fan-out — replacing three sequential
        //    fork-joins (one per qProj/kProj/vProj.forward call) with one.
        //    The runtime guarantees the reduction order is byte-identical
        //    to three separate `matmulXTQThreaded` calls, so this preserves
        //    the byte-exact llama.cpp match.
        //
        //    Fallback (F32 weight on any projection, or a runtime gate
        //    miss leaving the fused result as nulls): three independent
        //    consumers of x → clone twice; last use moves the original.
        var qRaw:Tensor;
        var kRaw:Tensor;
        var v:Tensor;
        var qWq = qProj.qweight;
        var kWq = kProj.qweight;
        var vWq = vProj.qweight;
        // Each projection is `matmul → reshape`. The matmul produces a
        // fresh tensor; `.reshape` then returns a VIEW of it whose refcount
        // is independent from the matmul's own ref. Without explicit free
        // the matmul output's storage leaks (the reshape view eventually
        // freed releases one ref, but the original ref to the underlying
        // buffer is never decremented). Capture the matmul into a local,
        // reshape, then drop the local — leaves only the view refcount on
        // the storage.
        if (qWq != null && kWq != null && vWq != null) {
            var triple = QTensor.fusedQkvMatmul(x, qWq, kWq, vWq, 0);
            if (triple != null && triple.length == 3
                && triple[0] != null && triple[1] != null && triple[2] != null) {
                qRaw = triple[0].reshape([seqQ, numQHeads, headDim]);
                triple[0].free();
                kRaw = triple[1].reshape([seqQ, numKvHeads, headDim]);
                triple[1].free();
                v    = triple[2].reshape([seqQ, numKvHeads, headDim]);
                triple[2].free();
            } else {
                // Gate-miss inside the kernel (SDOT unavailable, batch != 1,
                // x non-contiguous, …): fall back to the three-call path.
                var qProjOut = qProj.forward(x.clone());
                qRaw = qProjOut.reshape([seqQ, numQHeads, headDim]);
                qProjOut.free();
                var kProjOut = kProj.forward(x.clone());
                kRaw = kProjOut.reshape([seqQ, numKvHeads, headDim]);
                kProjOut.free();
                var vProjOut = vProj.forward(x);
                v    = vProjOut.reshape([seqQ, numKvHeads, headDim]);
                vProjOut.free();
            }
        } else {
            var qProjOut = qProj.forward(x.clone());
            qRaw = qProjOut.reshape([seqQ, numQHeads, headDim]);
            qProjOut.free();
            var kProjOut = kProj.forward(x.clone());
            kRaw = kProjOut.reshape([seqQ, numKvHeads, headDim]);
            kProjOut.free();
            var vProjOut = vProj.forward(x);
            v    = vProjOut.reshape([seqQ, numKvHeads, headDim]);
            vProjOut.free();
        }

        // 2) Rotary embedding — rotates the absolute positions starting
        //    at `positionOffset` (so the new tokens line up with cache).
        var q = rope.apply(qRaw, positionOffset);
        var k = rope.apply(kRaw, positionOffset);

        // 3) Push the new K/V into the cache. Subsequent reads use the
        //    full active slice (prior tokens + just-added).
        cache.append(k, v);
        var kAll = cache.keysView();   // [cache.currentLen, numKvHeads, headDim]
        var vAll = cache.valuesView(); // same shape

        // 4-7) Decode-step fast path: fused attention kernel.
        //
        // The bmm + scale + mask + softmax + bmm chain below materialises
        // `expandKvHeads(kAll)` + `expandKvHeads(vAll)` (~147 MB/token at
        // 16 layers, cacheLen=568) plus three score tensors. For the
        // single-query decode case (`seqQ == 1`) every causal-mask cell
        // is visible (the cache only contains tokens up to the current
        // position), so we can skip the mask entirely and stream over the
        // un-expanded KV cache once. `flashAttnDecode` returns the same
        // [1, numQHeads, headDim] result as the unfused path; on any
        // gate violation (non-F32, non-contig Q, GQA group mismatch,
        // unexpected strides) it returns null and we fall through to the
        // bmm chain.
        if (seqQ == 1) {
            var ctx = q.flashAttnDecode(kAll, vAll, scale);
            if (ctx != null) {
                // ctx is [1, numQHeads, headDim] contiguous owning.
                // reshape collapses to [seqQ, hidden] (a view; the data
                // layout already matches contiguous row-major).
                var hiddenSize = numQHeads * headDim;
                var ctxFlat = ctx.reshape([seqQ, hiddenSize]);
                var out = oProj.forward(ctxFlat);
                ctxFlat.free();
                ctx.free();
                return out;
            }
            // Fall-through to the unfused path on gate failure.
        }

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
        var scoresRaw = qByHead.bmmThreaded(kT, 0);  // [numQHeads, seqQ, cacheLen]
        var scoresScaled = scoresRaw.scale(scale);
        scoresRaw.free();

        // 5) Causal mask. Each query row i can only see keys
        //    [0, i + positionOffset]. This shifts the mask diagonal by
        //    positionOffset. NB: `applyMask` (causalMask_) is IN-PLACE
        //    and returns the same handle — scoresMasked == scoresScaled.
        var scoresMasked = applyMask(scoresScaled, positionOffset);

        // 6) Softmax along the last dim (the key axis). Allocates fresh;
        //    scoresMasked (== scoresScaled) is dead.
        var attn = scoresMasked.softmax();             // [numQHeads, seqQ, cacheLen]
        scoresMasked.free();

        // 7) Context = attn @ V, per head.
        var context = attn.bmmThreaded(vAllExpanded, 0);  // [numQHeads, seqQ, headDim]
        attn.free();

        // 8) Bring heads back to the row axis and flatten head/head_dim.
        //    `reshape` after `permute` materialises a fresh contiguous
        //    tensor (non-contiguous source path), so contextFlat is a
        //    new allocation and contextRowMajor can be released alongside
        //    the underlying context.
        var contextRowMajor = context.permute([1, 0, 2]); // [seqQ, numQHeads, headDim]
        var hiddenSize = numQHeads * headDim;
        var contextFlat = contextRowMajor.reshape([seqQ, hiddenSize]);

        // 9) Output projection back to hidden_size.
        var out = oProj.forward(contextFlat);

        // Manual free of the per-layer transients. The compiler's
        // InsertFreePass doesn't know about `rayzor_tensor_free`, so
        // the tensor refcount machinery only fires for explicit `.free()`
        // calls inserted here. Without these every decoded token leaks
        // ~17 MB of attention scratch × 16 layers, climbing to ~28 GB
        // resident at 500 tokens before macOS swap pressure makes the
        // system feel frozen. Frees run AFTER `oProj.forward` so any
        // worker threads spawned for the threaded path have rejoined.
        kAllExpanded.free();
        vAllExpanded.free();
        contextFlat.free();
        contextRowMajor.free();
        context.free();

        return out;
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
        // Strided gather + memcpy per (qh, j). Replaces the per-element
        // triple-loop that did ~2 * num_q_heads * seqK * head_dim extern
        // calls per invocation (~40% of decode time at typical Llama 3.2
        // 1B shapes). The runtime kernel is contiguous-innermost-only;
        // KVCache's slice() views preserve contiguity along axis 2 so the
        // fast path fires. Falls back to the original triple-loop if the
        // primitive returns null (non-F32 / non-contiguous innermost).
        var group = Std.int(numQHeads / numKvHeads);
        var out = t.expandKvHeadsAxis1(group);
        if (out != null) {
            return out;
        }
        var s = t.shape();
        var seqK = s[0];
        var headDimV = s[2];
        var fallback = Tensor.zeros([numQHeads, seqK, headDimV], F32);
        for (qh in 0...numQHeads) {
            var kvh = Std.int(qh / group);
            for (j in 0...seqK) {
                for (d in 0...headDimV) {
                    fallback.set([qh, j, d], t.get([j, kvh, d]));
                }
            }
        }
        return fallback;
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
