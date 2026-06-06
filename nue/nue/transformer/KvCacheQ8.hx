package nue.transformer;

import rayzor.ds.Tensor;

/**
 * Q8_0 KV cache — opaque extern class delivered by the `nue-plugins`
 * native plugin (`libnue_plugins.dylib`), loaded via the consumer
 * project's `rayzor.toml` `[build] native-libs` entry.
 *
 * Storage per (row, kv_head) is `head_dim / 32` Q8_0 blocks of 34
 * bytes (2-byte f16 scale + 32 i8 quants). ~3.76× smaller than F32
 * — meaningful when the KV cache dominates working set on long
 * contexts.
 *
 * Used by [KVCache] when the head dim is a multiple of 32 and the
 * `RAYZOR_KV_Q8=1` opt-in is set.
 */
@:keep
extern class KvCacheQ8 {
    /**
     * Allocate a fresh Q8_0 KV cache. Returns null on invalid
     * arguments (non-positive sizes, head_dim not a multiple of 32,
     * malloc failure). Caller owns the handle and must call `free()`.
     */
    public static function alloc(maxSeqLen:Int, numKvHeads:Int, headDim:Int):KvCacheQ8;

    /** Release the underlying buffer. Idempotent on null/zero handle. */
    public function free():Void;

    /**
     * Quantise + append `newRows` (F32, shape `[n, num_kv_heads,
     * head_dim]`) at write cursor `currentLen`. Returns the new
     * write cursor, or `-1` on shape/dtype mismatch or overflow.
     */
    public function append(currentLen:Int, newRows:Tensor):Int;

    /**
     * Dequantise positions `[0..currentLen)` into a fresh owning F32
     * tensor of shape `[currentLen, num_kv_heads, head_dim]`. Used
     * by the prefill BMM path that still needs F32 K/V. Returns null
     * on overflow.
     */
    public function dequantView(currentLen:Int):Tensor;

    /**
     * Fused single-pass flash attention over a Q8 KV cache. Self is
     * the K cache; pass V cache + Q tensor + cache length + group
     * head count + softmax scale. Returns a fresh F32 tensor of
     * shape `[1, num_q_heads, head_dim]`.
     */
    public function flashAttnDecodeQ8(
        q:Tensor, vCache:KvCacheQ8, cacheLen:Int, numQHeads:Int, scale:Float
    ):Tensor;
}
