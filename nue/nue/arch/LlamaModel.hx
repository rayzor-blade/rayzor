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
        // The reassign-then-pass-self pattern (`x = f(x)`) is
        // semantically a move-and-rebind, but the cross-file strict
        // @:move analyzer flags the call-arg read as a use-after-move.
        // Pass the clone into the block forward so the binding `h` is
        // only ever read as a method receiver — never as a direct
        // `Variable` call argument — which the analyzer treats as a
        // use rather than a move.
        var h = embedTokens.lookup(tokenIds);
        for (block in blocks) {
            var hIn = h.clone();
            h = block.forward(hIn);
        }
        var hNormIn = h.clone();
        var normed = outputNorm.forward(hNormIn);
        return lmHead.forward(normed);
    }

    public function forward(x:Tensor):Tensor {
        var shape = x.shape();
        var n = shape[0];
        for (i in 1...shape.length) n *= shape[i];
        var ids = [];
        for (i in 0...n) ids.push(Std.int(x.get([i])));
        return forwardIds(ids);
    }

    public function resetCache():Void {
        // Concrete downcasts, not `Dynamic` field access. The blocks are
        // held as Array<Module>, so each element arrives here as an
        // interface fat pointer (`obj_ptr @ offset 0, fn_ptr_N @ offset
        // 8(N+1)`). Dynamic field access reads offset 0 as a class
        // type-id header — for a fat pointer that's the inner obj_ptr's
        // address truncated to i64, which lands on a class id that
        // either doesn't exist or maps to an unrelated class. The
        // class→class cast lowering unwraps the fat pointer correctly.
        for (i in 0...blocks.length) {
            var tb = cast(blocks[i], TransformerBlock);
            var attn = cast(tb.attn, GQAttention);
            if (attn != null && attn.cache != null) {
                attn.cache.reset();
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
