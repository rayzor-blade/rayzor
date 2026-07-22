package nue.engine;

import rayzor.Usize;

/**
 * BNNSGraph whole-encoder engine (BERT Phase 4) — macOS only, no-op elsewhere.
 *
 * A nue engine, not a stdlib primitive: the CoreML runtime + FFI symbols are
 * provided by the nue-plugins dylib (alongside the Q8 KV cache + flash
 * kernels), but the model-specific encoder plumbing is nue's.
 *
 * MODEL-GENERIC across the BERT family: `load` takes the artifact directory
 * plus the model's gguf STEM (artifacts are `<stem>.encoder_s{S}.mlmodelc`,
 * authored from that same gguf by `bert_graph_author.py`) and the model's
 * hidden size, and returns a HANDLE (> 0) that scopes every later call — so
 * multiple BERT models can be live in one process without colliding.
 *
 * One `execute` call runs the whole fused encoder: post-embedding hidden
 * states `[S, hidden]` + additive key-mask bias `[S]` (0 real / -1e4 pad —
 * fp16-safe) → final hidden states `[S, hidden]`.
 *
 * Buffers cross as raw `Usize` addresses; strings as `Bytes.ofString`
 * address + byte length (caller owns and frees the Bytes).
 */
extern class BertGraph {
    /** Handle > 0 on success, 0 when no artifacts found, -1 off-Mac. */
    @:native("nue_bert_graph_load")
    static function load(dirPtr:Usize, dirLen:Int, stemPtr:Usize, stemLen:Int, hidden:Int, kind:Int):Int;

    /** Smallest loaded bucket of `handle` that fits `seq`, or 0. */
    @:native("nue_bert_graph_bucket")
    static function bucketFor(handle:Int, seq:Int):Int;

    /** Run bucket `s`: h[s*hidden] + bias[s] → out[s*hidden], f32. 0 = ok. */
    @:native("nue_bert_graph_execute")
    static function execute(handle:Int, s:Int, h:Usize, bias:Usize, out:Usize):Int;
}
