package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import nue.transformer.KvCacheQ8;

/**
 * Per-layer key/value cache for incremental decoding.
 *
 * Maintains two append-only buffers of shape
 * `[max_seq_len, num_kv_heads, head_dim]` and tracks the current write
 * position. During the prefill phase the entire prompt is written in one
 * shot via `append`; during decode each new token's K and V are appended
 * one row at a time.
 *
 * The "active" portion of the cache (positions 0..currentLen) is exposed
 * via `keysView()` / `valuesView()` as a no-copy slice. Attention then
 * reads those slices as the K, V it scores against.
 *
 * Memory model: we allocate `max_seq_len` rows up front. For a typical
 * Llama 3.2 1B inference run (16 layers × 8 KV heads × 64 head_dim × 2
 * (k+v) × 2 bytes (F16) × 2048 tokens) this is ~16 MB of cache total —
 * cheap on native, the tight side on WebGPU where per-buffer caps may
 * require splitting across layers (handled at the model level).
 */
class KVCache {
    // F32 backing — populated only when `useQ8` is false.
    public var keys:Tensor;
    public var values:Tensor;
    // Q8_0 backing — populated only when `useQ8` is true. Storage is
    // ~3.76× smaller than F32; dequant happens on read (prefill bmm) or
    // is fully fused (decode flash kernel).
    public var keysQ8:KvCacheQ8;
    public var valuesQ8:KvCacheQ8;
    public var useQ8:Bool;
    public var currentLen:Int;
    public var maxSeqLen:Int;
    public var numKvHeads:Int;
    public var headDim:Int;
    public var dtype:DType;

    public function new(
        maxSeqLen:Int, numKvHeads:Int, headDim:Int, dtype:DType,
        useQ8:Bool = false
    ) {
        this.maxSeqLen = maxSeqLen;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.dtype = dtype;
        this.currentLen = 0;
        // Q8_0 super-block is 32 elements; head_dim must be a multiple
        // of 32 or we fall back to F32. (Llama 3.2 1B's head_dim=64 is
        // fine; Qwen 0.5B's 64 too; Mistral 7B's 128 too — every
        // mainstream LLM checkpoint satisfies this.)
        var canQ8 = useQ8 && (headDim % 32 == 0);
        if (canQ8) {
            this.keysQ8 = KvCacheQ8.alloc(maxSeqLen, numKvHeads, headDim);
            this.valuesQ8 = KvCacheQ8.alloc(maxSeqLen, numKvHeads, headDim);
            if (this.keysQ8 == null || this.valuesQ8 == null) {
                // alloc failure — release whichever side did succeed
                // and fall through to the F32 path.
                if (this.keysQ8 != null) { this.keysQ8.free(); this.keysQ8 = null; }
                if (this.valuesQ8 != null) { this.valuesQ8.free(); this.valuesQ8 = null; }
                canQ8 = false;
            }
        }
        this.useQ8 = canQ8;
        if (!canQ8) {
            // F32 path: pre-allocate full capacity. Zero-init is fine
            // because the active region (currentLen) bounds what
            // attention reads.
            this.keys = Tensor.zeros([maxSeqLen, numKvHeads, headDim], dtype);
            this.values = Tensor.zeros([maxSeqLen, numKvHeads, headDim], dtype);
        }
    }

    /**
     * Append `n` new tokens' K and V to the cache. `newKeys` and
     * `newValues` are tensors of shape `[n, num_kv_heads, head_dim]`.
     * Returns the new currentLen.
     *
     * F32 path: `appendAlong0` — single bulk-row memcpy.
     * Q8 path: quantise-on-write per-block — fold dtype conversion
     *          into the same pass over `newRows`.
     *
     * On failure (dtype/shape/OOB) we `throw` rather than silently
     * mask the bug.
     */
    public function append(newKeys:Tensor, newValues:Tensor):Int {
        var n = newKeys.shape()[0];
        if (currentLen + n > maxSeqLen) {
            throw "KVCache overflow: " + (currentLen + n) + " > " + maxSeqLen;
        }
        if (useQ8) {
            var newK = keysQ8.append(currentLen, newKeys);
            var newV = valuesQ8.append(currentLen, newValues);
            if (newK < 0 || newV < 0) {
                throw "KVCacheQ8.append failed (k=" + newK + ", v=" + newV + ")";
            }
            currentLen = newK;
        } else {
            var krc = keys.appendAlong0(newKeys, currentLen);
            var vrc = values.appendAlong0(newValues, currentLen);
            if (krc != 0 || vrc != 0) {
                throw "KVCache.append failed (krc=" + krc + ", vrc=" + vrc + ")";
            }
            currentLen += n;
        }
        return currentLen;
    }

    /**
     * Active K slice as an F32 tensor of shape `[currentLen,
     * num_kv_heads, head_dim]`.
     *
     * F32 path returns a no-copy VIEW (shares storage with the cache);
     * Q8 path dequantises into a fresh OWNING tensor that the caller
     * must `.free()`. Both shapes are equivalent for downstream use;
     * the prefill bmm chain frees its kAll/vAll explicitly after
     * `expandKvHeads` so the contract is honoured in either mode.
     *
     * The decode fast path in GQAttention skips this entirely and
     * dispatches to `keysQ8.flashAttnDecodeQ8` directly, avoiding the
     * dequant allocation.
     */
    public function keysView():Tensor {
        if (useQ8) {
            return keysQ8.dequantView(currentLen);
        }
        return keys.slice(0, 0, currentLen);
    }

    /** Active V slice — counterpart to `keysView` with the same ownership rules. */
    public function valuesView():Tensor {
        if (useQ8) {
            return valuesQ8.dequantView(currentLen);
        }
        return values.slice(0, 0, currentLen);
    }

    /** Reset to empty (start a new conversation / prompt). */
    public function reset():Void {
        currentLen = 0;
    }
}
