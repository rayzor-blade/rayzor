package nue.arch;

import nue.Module;
import nue.CausalLanguageModel;
import nue.Embedding;
import nue.Linear;
import nue.transformer.RoPE;
import nue.transformer.RMSNorm;
import nue.transformer.GQAttention;
import nue.transformer.LlamaBlock;
import nue.engine.PrefillGraph;
import rayzor.Bytes;
import nue.model.ModelMetadata;
import rayzor.ds.Tensor;
import rayzor.ds.DType;

/**
 * Concrete `LanguageModel` for the Llama family. Constructed by
 * `LlamaArch.build()` — not normally instantiated directly.
 *
 * Architecture:
 * ```
 *   tokens [seq_len]
 *     ↓ embedTokens
 *   x [seq_len, hidden_size]
 *     ↓ blocks[0..N-1].forward     (each is a TransformerBlock)
 *   h [seq_len, hidden_size]
 *     ↓ outputNorm                  (RMSNorm)
 *   h' [seq_len, hidden_size]
 *     ↓ lmHead                      (Linear; weight-tied with embed if config says)
 *   logits [seq_len, vocab_size]
 * ```
 */
class LlamaModel implements CausalLanguageModel {
    /** Spin pool shared by the quant Linears (pure-Haxe matmul); joined by
        shutdownPool(). Null when the FFI matmul path is active. */
    public var spinPool:Null<rayzor.concurrent.SpinPool> = null;

    /** Join the matmul pool's workers. Call before process exit — the
        runtime waits on all live threads before JIT teardown. */
    public function shutdownPool():Void {
        if (spinPool != null) {
            spinPool.shutdown();
            spinPool = null;
        }
    }

    public var metadata:ModelMetadata;
    public var embedTokens:Embedding;
    public var blocks:Array<LlamaBlock>;
    public var outputNorm:RMSNorm;
    public var lmHead:Linear;
    public var sharedRope:RoPE;
    /** Fused CoreML prefill engine handle (`PrefillGraph.load`, set by the
        loader when artifacts sit next to the gguf; 0 = off). Prefill runs
        the whole block stack in one graph call and seeds every layer's KV
        cache from the graph outputs; decode continues on the CPU path. */
    public var prefillHandle:Int = 0;

    public function new(
        metadata:ModelMetadata,
        embedTokens:Embedding,
        blocks:Array<LlamaBlock>,
        outputNorm:RMSNorm,
        lmHead:Linear,
        sharedRope:RoPE
    ) {
        this.metadata = metadata;
        this.embedTokens = embedTokens;
        this.blocks = blocks;
        this.outputNorm = outputNorm;
        this.lmHead = lmHead;
        this.sharedRope = sharedRope;
    }

    public function forwardIds(tokenIds:Array<Int>):Tensor {
        // Fused-graph prefill (see forwardLastLogits). Returns the LAST row's
        // logits [1, vocab] — every caller reduces to lastRow anyway.
        if (prefillHandle > 0 && cacheLen() == 0) {
            var g = graphPrefill(tokenIds);
            if (g != null) return g;
        }
        var h = embedTokens.lookup(tokenIds);
        for (block in blocks) {
            h = block.forward(h);
        }
        var normed = outputNorm.forward(h);
        var result = lmHead.forward(normed);
        normed.free();
        h.free();
        return result;
    }

    public function forwardLastLogits(tokenIds:Array<Int>):Tensor {
        // Fused-graph prefill: one CoreML call for the whole block stack,
        // KV caches seeded from the graph outputs (~15x the per-op path).
        // Only from an empty cache; falls through to the per-op path on any
        // failure or when no bucket fits.
        if (prefillHandle > 0 && cacheLen() == 0) {
            var g = graphPrefill(tokenIds);
            if (g != null) return g;
        }
        var h = embedTokens.lookup(tokenIds);
        for (block in blocks) {
            h = block.forward(h);
        }
        var seqLen = h.shape()[0];
        if (seqLen > 1) {
            var last = h.slice(0, seqLen - 1, seqLen);
            var normed = outputNorm.forward(last);
            var result = lmHead.forward(normed);
            normed.free();
            last.free();
            h.free();
            if (dbgLogit) dumpArgmax(result);
            return result;
        }
        var normed1 = outputNorm.forward(h);
        var result1 = lmHead.forward(normed1);
        normed1.free();
        h.free();
        if (dbgLogit) dumpArgmax(result1);
        return result1;
    }

    static var dbgLogit:Bool = Sys.getEnvOr("NUE_DBG_LOGIT", "") == "1";

    static function dumpArgmax(logits:Tensor):Void {
        var n = logits.numel();
        if (n <= 0) return;
        var best = 0;
        var bv = logits.getFlat(0);
        for (i in 1...n) {
            var v = logits.getFlat(i);
            if (v > bv) { bv = v; best = i; }
        }
        Sys.println("[logit] argmax=" + best + " max=" + bv + " vocab=" + n);
    }

    /** Warm the forward + DECODE kernels at load so the first request runs hot.
        The bundle tier-promotes by call count, so an unwarmed first request
        promotes DURING its own generation — slow, low-floor. This warms both
        decode entry points (forwardIds default; forwardLastLogits for the
        NUE_PREFILL_LAST_LOGITS=1 / server path) with ~40 single-token forwards
        each, clearing the WARM/HOT thresholds.

        Runs for ANY model, INDEPENDENT of the CoreML graph — decode is a
        pure-CPU, bandwidth-bound path either way. With the graph on, the seeding
        prefill also pays the one-time E5RT specialization; with it off
        (NUE_PREFILL=off, pure-CPU per-op prefill) the warm is cheap (~1s). The
        loader gates this via NUE_DECODE_WARM (opt-out), so a no-CoreML server
        still gets a warm decode from request 1. */
    public function warm():Void {
        // forwardIds path (default): prefill (graph or per-op) seeds the cache,
        // then single-token forwards warm + tier-promote the per-op decode.
        var a = forwardIds([0]);
        if (a != null) a.free();
        for (i in 0...40) {
            var d = forwardIds([1]);
            if (d != null) d.free();
        }
        resetCache();
        // forwardLastLogits path (server, NUE_PREFILL_LAST_LOGITS=1): same wrapper
        // drives both its prefill and its per-token decode, so warm it too.
        var b = forwardLastLogits([0]);
        if (b != null) b.free();
        for (i in 0...40) {
            var e = forwardLastLogits([1]);
            if (e != null) e.free();
        }
        resetCache();
    }

    /** Graph prefill: pad ids to the bucket (causal masking keeps pad rows
        self-contained; only the real-prompt prefix rows are appended to the
        caches), run the fused graph, seed every layer's KV, and return the
        last real row's logits. Null = not applicable, caller falls through. */
    function graphPrefill(tokenIds:Array<Int>):Null<Tensor> {
        var sReal = tokenIds.length;
        var bucket:Int = PrefillGraph.bucketFor(prefillHandle, sReal);
        if (bucket <= 0) return null;
        var padded = tokenIds.copy();
        while (padded.length < bucket) padded.push(0);
        var h = embedTokens.lookup(padded);
        var hs = metadata.hiddenSize;
        var kvh = metadata.numKvHeads;
        var hd = metadata.headDim;
        var out = Tensor.uninit([bucket, hs], DType.F32);
        var kvB = Bytes.alloc(metadata.numLayers * 2 * bucket * kvh * hd * 4);
        var rc:Int = PrefillGraph.execute(prefillHandle, bucket, h.data().raw(), out.data().raw(), kvB.address());
        h.free();
        if (rc != 0) {
            kvB.free();
            out.free();
            return null;
        }
        for (i in 0...blocks.length) {
            var kT = Tensor.uninit([sReal, kvh, hd], DType.F32);
            var vT = Tensor.uninit([sReal, kvh, hd], DType.F32);
            PrefillGraph.kvCopy(kvB.address(), 2 * i, sReal, bucket, kvh, hd, kT.data().raw());
            PrefillGraph.kvCopy(kvB.address(), 2 * i + 1, sReal, bucket, kvh, hd, vT.data().raw());
            blocks[i].attn.cache.append(kT, vT);
            kT.free();
            vT.free();
        }
        kvB.free();
        var last = out.slice(0, sReal - 1, sReal);
        var normed = outputNorm.forward(last);
        var result = lmHead.forward(normed);
        normed.free();
        last.free();
        out.free();
        return result;
    }

    public function forward(x:Tensor):Tensor {
        var shape = x.shape();
        var n = shape[0];
        for (i in 1...shape.length) n *= shape[i];
        var ids = [];
        for (i in 0...n) ids.push(Std.int(x.getFlat(i)));
        return forwardIds(ids);
    }

    public function resetCache():Void {
        for (i in 0...blocks.length) {
            var attn = blocks[i].attn;
            if (attn != null && attn.cache != null) {
                attn.cache.reset();
            }
        }
    }

    public function cacheLen():Int {
        if (blocks.length == 0) return 0;
        var attn = blocks[0].attn;
        if (attn == null || attn.cache == null) return 0;
        return attn.cache.currentLen;
    }

    public function rewindCache(len:Int):Void {
        for (i in 0...blocks.length) {
            var attn = blocks[i].attn;
            if (attn != null && attn.cache != null) {
                attn.cache.rewind(len);
            }
        }
    }

    public function parameters():Array<nue.Module.NamedTensor> {
        // Llama GGUF inference builds concrete tensors up front through the
        // loader, and does not need recursive parameter introspection at run
        // time. Keeping this inert avoids the current compiler resolver bug
        // around `NamedTensor.name` typedef fields, which otherwise leaves a
        // SIGILL stub in release bundles.
        return [];
    }
}
