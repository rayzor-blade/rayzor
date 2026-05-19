package nue.arch;

import nue.Module;
import nue.CausalLanguageModel;
import nue.Embedding;
import nue.Linear;
import nue.transformer.RoPE;
import nue.transformer.RMSNorm;
import nue.transformer.TransformerBlock;
import nue.transformer.GQAttention;
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
    public var metadata:ModelMetadata;
    public var embedTokens:Embedding;
    public var blocks:Array<Module>;
    public var outputNorm:RMSNorm;
    public var lmHead:Linear;
    public var sharedRope:RoPE;

    public function new(
        metadata:ModelMetadata,
        embedTokens:Embedding,
        blocks:Array<Module>,
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
        var x = embedTokens.lookup(tokenIds);
        for (block in blocks) {
            x = block.forward(x);
        }
        x = outputNorm.forward(x);
        return lmHead.forward(x);
    }

    public function forward(x:Tensor):Tensor {
        var n = x.numel();
        var ids = [];
        for (i in 0...n) ids.push(Std.int(x.get([i])));
        return forwardIds(ids);
    }

    public function resetCache():Void {
        for (block in blocks) {
            // The cache is the GQAttention.cache field — reach in via
            // a downcast since the generic TransformerBlock holds the
            // attention as a `Module`.
            var tb:Dynamic = block;
            if (tb.attn != null && tb.attn.cache != null) {
                tb.attn.cache.reset();
            }
        }
    }

    public function parameters():Array<nue.Module.NamedTensor> {
        var ps = [];
        for (p in embedTokens.parameters())
            ps.push({ name: "token_embd." + p.name, tensor: p.tensor });
        for (block in blocks) {
            for (p in block.parameters()) ps.push(p);
        }
        for (p in outputNorm.parameters())
            ps.push({ name: "output_norm." + p.name, tensor: p.tensor });
        if (!metadata.tieWordEmbeddings) {
            for (p in lmHead.parameters())
                ps.push({ name: "output." + p.name, tensor: p.tensor });
        }
        return ps;
    }
}
