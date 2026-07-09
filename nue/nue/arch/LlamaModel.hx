package nue.arch;

import nue.Module;
import nue.CausalLanguageModel;
import nue.Embedding;
import nue.Linear;
import nue.transformer.RoPE;
import nue.transformer.RMSNorm;
import nue.transformer.GQAttention;
import nue.transformer.LlamaBlock;
import nue.model.ModelMetadata;
import rayzor.ds.Tensor;

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
            return result;
        }
        var normed1 = outputNorm.forward(h);
        var result1 = lmHead.forward(normed1);
        normed1.free();
        h.free();
        return result1;
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
