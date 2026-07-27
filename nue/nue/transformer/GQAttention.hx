package nue.transformer;

import nue.Module;
import nue.Linear;
import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
import rayzor.ds.DType;
import nue.transformer.FlashDecode;
import nue.transformer.Q8Cache;

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
 * attention loop — see `expandKvHeads` below. The math is the standard
 * scaled-dot-product pattern; the inner matmuls do all the work.
 *
 * Choices left for the caller to override by subclassing:
 *   - alibi bias, sliding window mask: re-implement `applyMask`
 *   - flash-attention-style streaming: re-implement `forward` to fuse
 *     across heads / sub-blocks
 *   - MoE routing across heads: re-implement `forward` for sparse
 *     per-token expert selection
 */
class GQAttention implements Attention {
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
        //    Fast path: when all three projections are quantized, dispatch
        //    through the fused QKV kernel — x is quantized to Q8_K once and
        //    shared across all three weights in a single fan-out, with a
        //    reduction order identical to three separate matmuls.
        //
        //    Fallback (an F32 weight on any projection, or a kernel gate
        //    miss returning nulls): three independent Linear calls.
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
        var useHaxeMat = Linear.useHaxeMatmul();
        // Row-wise (INT8/Q8_0) triples fuse by default — a separate, cheaper
        // mechanism than the opt-in k-quant row-space fusion: only the shared
        // activation quantise is hoisted, so Q/K/V quantise the post-attn-norm
        // activation ONCE instead of three times. Bit-identical output.
        var useFusedMat = !useHaxeMat || nue.Q4Matmul.useFusedMatmul()
            || nue.Q4Matmul.canFuseRowwise(qWq, kWq, vWq);
        if (qWq != null && kWq != null && vWq != null && useFusedMat) {
            var triple:Array<Tensor>;
            if (Linear.useHaxeMatmul()) {
                triple = nue.Q4Matmul.matmulFused(qWq, kWq, vWq, x, qProj.pool);
            } else {
                // Slots stay null on kernel gate-miss; the null-check below
                // falls back to the three-call path.
                triple = [null, null, null];
                QTensor.fusedQkvIntoArr(x, qWq, kWq, vWq, 0, triple);
            }
            if (triple != null && triple.length == 3
                && triple[0] != null && triple[1] != null && triple[2] != null) {
                // The fused QKV kernel multiplies the raw quant weights and
                // does NOT go through Linear.forward, so the projection bias
                // (Qwen2 q/k/v) must be applied here — otherwise it's silently
                // dropped and attention is garbage. No-op for Llama (null bias).
                if (qProj.bias != null) triple[0].addInto(qProj.bias);
                qRaw = triple[0].reshape([seqQ, numQHeads, headDim]);
                triple[0].free();
                if (kProj.bias != null) triple[1].addInto(kProj.bias);
                kRaw = triple[1].reshape([seqQ, numKvHeads, headDim]);
                triple[1].free();
                if (vProj.bias != null) triple[2].addInto(vProj.bias);
                v    = triple[2].reshape([seqQ, numKvHeads, headDim]);
                triple[2].free();
            } else {
                // Gate-miss inside the kernel (SDOT unavailable, batch != 1,
                // x non-contiguous, …): fall back to the three-call path.
                // Linear.forward does NOT free its parameter (caller-frees
                // convention), so bind each clone and free it after the call
                // — an anonymous `x.clone()` argument leaks its refcount.
                var xq = x.clone();
                var qProjOut = qProj.forward(xq);
                xq.free();
                qRaw = qProjOut.reshape([seqQ, numQHeads, headDim]);
                qProjOut.free();
                var xk = x.clone();
                var kProjOut = kProj.forward(xk);
                xk.free();
                kRaw = kProjOut.reshape([seqQ, numKvHeads, headDim]);
                kProjOut.free();
                var vProjOut = vProj.forward(x);
                v    = vProjOut.reshape([seqQ, numKvHeads, headDim]);
                vProjOut.free();
            }
        } else {
            var xq = x.clone();
            var qProjOut = qProj.forward(xq);
            xq.free();
            qRaw = qProjOut.reshape([seqQ, numQHeads, headDim]);
            qProjOut.free();
            var xk = x.clone();
            var kProjOut = kProj.forward(xk);
            xk.free();
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
        // rope() allocates fresh output tensors — the pre-rotation
        // projections are dead from here. Without these two frees every
        // decoded token leaked 2 tensors × 16 layers (part of the known
        // ~8 MB/token decode-loop leak).
        qRaw.free();
        kRaw.free();

        // 3) Push the new K/V into the cache. Subsequent reads use the
        //    full active slice (prior tokens + just-added).
        cache.append(k, v);
        // append() copies rows into the cache's own storage (F32
        // appendAlong0 or Q8 quantise-on-write) — k and v are dead.
        k.free();
        v.free();

        // 4-7) Decode-step fast path: fused attention kernel.
        //
        // The bmm + scale + mask + softmax + bmm chain below materialises
        // `expandKvHeads(kAll)` + `expandKvHeads(vAll)` (~147 MB/token at
        // 16 layers, cacheLen=568) plus three score tensors. For the
        // single-query decode case (`seqQ == 1`) every causal-mask cell
        // is visible (the cache only contains tokens up to the current
        // position), so we can skip the mask entirely and stream over the
        // un-expanded KV cache once.
        //
        // Q8 path: dispatch directly to the Q8 fused kernel, skipping
        // the F32 dequant view. The kernel streams over Q8 K/V blocks,
        // dequantising into a 32-f32 stack buffer per block.
        // Pure-Haxe Q8 decode attention (guest-owned cache + SDOT kernel,
        // banded across kv-heads on the matmul pool).
        if (seqQ == 1 && cache.useQ8H && FlashDecode.enabled() && qProj.pool != null) {
            var ctx = FlashDecode.decode(
                cache.keysQ8H, cache.valuesQ8H, q, cache.currentLen,
                numQHeads, scale, qProj.pool
            );
            if (ctx != null) {
                var hiddenSize = numQHeads * headDim;
                var ctxFlat = ctx.reshape([seqQ, hiddenSize]);
                var out = oProj.forward(ctxFlat);
                ctxFlat.free();
                ctx.free();
                q.free();
                return out;
            }
        }
        if (seqQ > 1 && seqQ <= FlashDecode.batchMax()
            && cache.useQ8H && FlashDecode.enabled() && qProj.pool != null) {
            var ctx = FlashDecode.decodeBatch(
                cache.keysQ8H, cache.valuesQ8H, q, positionOffset, seqQ,
                numQHeads, scale, qProj.pool
            );
            if (ctx != null) {
                var hiddenSize = numQHeads * headDim;
                var ctxFlat = ctx.reshape([seqQ, hiddenSize]);
                var out = oProj.forward(ctxFlat);
                ctxFlat.free();
                ctx.free();
                q.free();
                return out;
            }
        }
        if (seqQ == 1 && cache.useQ8) {
            // Prefer the host-parallel kernel: on the wasm run target it bands
            // attention across the embedder's native worker pool (otherwise idle
            // during serial guest attention). Returns null off that path — a
            // browser/no-host build or any native target — so fall back to the
            // serial Q8 kernel, which has identical numerics.
            var ctx = cache.keysQ8.flashAttnDecodeQ8Host(
                q, cache.valuesQ8, cache.currentLen, numQHeads, scale
            );
            if (ctx == null) {
                ctx = cache.keysQ8.flashAttnDecodeQ8(
                    q, cache.valuesQ8, cache.currentLen, numQHeads, scale
                );
            }
            if (ctx != null) {
                var hiddenSize = numQHeads * headDim;
                var ctxFlat = ctx.reshape([seqQ, hiddenSize]);
                var out = oProj.forward(ctxFlat);
                ctxFlat.free();
                ctx.free();
                q.free();
                return out;
            }
        }

        var kAll = cache.keysView();   // [cache.currentLen, numKvHeads, headDim]
        var vAll = cache.valuesView(); // same shape
        if (seqQ == 1) {
            // F32 cache fast path: same fused kernel against the live view.
            // `flashAttnDecode` returns the same [1, numQHeads, headDim]
            // result as the unfused path; on any gate violation (non-F32,
            // non-contig Q, GQA group mismatch, unexpected strides) it
            // returns null and we fall through to the bmm chain.
            var ctx = q.flashAttnDecode(kAll, vAll, scale);
            if (ctx != null) {
                var hiddenSize = numQHeads * headDim;
                var ctxFlat = ctx.reshape([seqQ, hiddenSize]);
                var out = oProj.forward(ctxFlat);
                ctxFlat.free();
                ctx.free();
                kAll.free();
                vAll.free();
                q.free();
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
        // kAll/vAll may be views (F32 cache) or fresh dequant tensors
        // (Q8 cache). tensor.free() is refcount-aware: view free drops
        // only the view's own ref so the underlying cache storage
        // stays alive; owning free releases the dequant buffer.
        kAll.free();
        vAll.free();
        // qByHead/kT are views (permute / transposeLast2); freeing them
        // drops only their own ref, then q's owning ref releases the
        // rotated-Q storage.
        qByHead.free();
        kT.free();
        q.free();

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

    /**
     * `Attention` conformance. GQA is the causal decode path — padding is
     * handled by the causal mask + per-request KV cache, not an additive
     * key bias — so it ignores `attnBias` and runs the normal forward. Only
     * the bidirectional encoder (MultiHeadAttention) uses the bias.
     */
    public function forwardMasked(x:Tensor, attnBias:Tensor):Tensor {
        return forward(x);
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
