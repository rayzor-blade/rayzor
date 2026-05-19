package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

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
    public var keys:Tensor;
    public var values:Tensor;
    public var currentLen:Int;
    public var maxSeqLen:Int;
    public var numKvHeads:Int;
    public var headDim:Int;
    public var dtype:DType;

    public function new(maxSeqLen:Int, numKvHeads:Int, headDim:Int, dtype:DType) {
        this.maxSeqLen = maxSeqLen;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.dtype = dtype;
        this.currentLen = 0;
        // Pre-allocate full capacity. Zero-init is fine because the active
        // region (currentLen) bounds what attention reads.
        this.keys = Tensor.zeros([maxSeqLen, numKvHeads, headDim], dtype);
        this.values = Tensor.zeros([maxSeqLen, numKvHeads, headDim], dtype);
    }

    /**
     * Append `n` new tokens' K and V to the cache. `newKeys` and `newValues`
     * are tensors of shape `[n, num_kv_heads, head_dim]`. Returns the new
     * currentLen.
     *
     * Per-element copy into the pre-allocated slot. A future revision can
     * specialise this to a single slice-write once Tensor exposes a strided
     * in-place copy primitive.
     */
    public function append(newKeys:Tensor, newValues:Tensor):Int {
        var n = newKeys.shape()[0];
        if (currentLen + n > maxSeqLen) {
            throw "KVCache overflow: " + (currentLen + n) + " > " + maxSeqLen;
        }
        // Write rows currentLen..currentLen+n.
        for (i in 0...n) {
            for (h in 0...numKvHeads) {
                for (d in 0...headDim) {
                    keys.set([currentLen + i, h, d], newKeys.get([i, h, d]));
                    values.set([currentLen + i, h, d], newValues.get([i, h, d]));
                }
            }
        }
        currentLen += n;
        return currentLen;
    }

    /**
     * Active key slice — view of `keys[0..currentLen]` with shape
     * `[currentLen, num_kv_heads, head_dim]`. No copy; the returned tensor
     * shares storage with the cache.
     */
    public function keysView():Tensor {
        return keys.slice(0, 0, currentLen);
    }

    /** Active value slice — counterpart to `keysView`. */
    public function valuesView():Tensor {
        return values.slice(0, 0, currentLen);
    }

    /** Reset to empty (start a new conversation / prompt). */
    public function reset():Void {
        currentLen = 0;
    }
}
