package nue.engine;

import rayzor.Usize;

/**
 * Fused Llama PREFILL graph engine (macOS, CPU CoreML) — one call runs the
 * whole block stack over a prompt bucket and returns the final hidden states
 * plus every layer's K (post-rope, interleaved GGUF convention) and V in
 * cache layout, ~15x the per-op Q4 prefill (llama_prefill_probe.py).
 *
 * A nue engine, not a stdlib primitive: the CoreML runtime + FFI symbols are
 * provided by the nue-plugins dylib (alongside the Q8 KV cache + flash
 * kernels), but the model-specific block-stack/KV plumbing is nue's.
 *
 * Artifacts `<gguf-stem>.prefill_s{S}.mlmodelc` sit next to the gguf, authored
 * by `nue-plugins/examples/llama_prefill_author.py`. `execute` writes the KV
 * block as `[layers][2][bucket*kv_heads*head_dim]` f32 (k slot 2i, v slot 2i+1);
 * `kvCopy` carves one slot's real-prompt prefix into a caller-owned
 * `[s_real, kv_heads, head_dim]` F32 buffer for `KVCache.append`.
 */
extern class PrefillGraph {
    @:native("nue_prefill_graph_load")
    static function load(dirPtr:Usize, dirLen:Int, stemPtr:Usize, stemLen:Int,
        hidden:Int, layers:Int, kvHeads:Int, headDim:Int):Int;

    /** Smallest loaded bucket of `handle` that fits `seq`, or 0. */
    @:native("nue_prefill_graph_bucket")
    static function bucketFor(handle:Int, seq:Int):Int;

    /** Run bucket `s`: h[s*hidden] → out[s*hidden] + the KV block. 0 = ok. */
    @:native("nue_prefill_graph_execute")
    static function execute(handle:Int, s:Int, h:Usize, out:Usize, kv:Usize):Int;

    /** Copy KV slot `slot` (2*layer=K, 2*layer+1=V) from the `execute` KV block
        into a caller-owned `[sReal, kvHeads, headDim]` F32 buffer `dst`. */
    @:native("nue_prefill_graph_kv_copy")
    static function kvCopy(kv:Usize, slot:Int, sReal:Int, bucket:Int, kvHeads:Int, headDim:Int, dst:Usize):Int;
}
