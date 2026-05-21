package nue;

import nue.Embedding;
import nue.transformer.LayerNorm;
import nue.transformer.TransformerBlock;
import nue.model.ModelMetadata;
import rayzor.ds.Tensor;

/**
 * BERT-family encoder. Composes the input embedding stack (token +
 * position + optional segment, with a LayerNorm on the sum), N
 * pre-norm `TransformerBlock`s, and a final encoder LayerNorm.
 *
 * The forward pass takes token IDs and returns per-token hidden
 * states `[seq_len, hidden_size]`. Padding masking, segment IDs, and
 * any task heads (MLM, classification, embedding pooling) live in
 * caller code or future `nue.head.*` modules.
 *
 * Segment embeddings are optional — many BERT GGUFs omit them or set
 * them to zero for the single-segment case. When the loader doesn't
 * register `token_types.weight`, `segmentEmbed` is `null` and the
 * forward path skips the segment contribution.
 *
 * **Status:** v1 — encoder forward path is wired end-to-end against
 * the `LayerNorm` + `MultiHeadAttention` + `LayerNorm` + `GeluFFN`
 * stack. Padding-aware attention masking is a follow-up (today the
 * encoder attends to every position regardless of `attentionMask`).
 */
class BertModel implements EncoderModel {
    public var meta:ModelMetadata;
    public var tokenEmbed:Embedding;
    public var positionEmbed:Embedding;
    public var segmentEmbed:Null<Embedding>;
    public var embedNorm:Null<LayerNorm>;
    public var blocks:Array<TransformerBlock>;
    public var encoderNorm:Null<LayerNorm>;

    public function new(
        meta:ModelMetadata,
        tokenEmbed:Embedding,
        positionEmbed:Embedding,
        ?segmentEmbed:Embedding,
        ?embedNorm:LayerNorm,
        blocks:Array<TransformerBlock>,
        ?encoderNorm:LayerNorm
    ) {
        this.meta = meta;
        this.tokenEmbed = tokenEmbed;
        this.positionEmbed = positionEmbed;
        this.segmentEmbed = segmentEmbed;
        this.embedNorm = embedNorm;
        this.blocks = blocks;
        this.encoderNorm = encoderNorm;
    }

    public function encode(tokenIds:Array<Int>, ?attentionMask:Array<Int>):Tensor {
        var seq = tokenIds.length;
        var positions = [for (i in 0...seq) i];

        var x = tokenEmbed.lookup(tokenIds);
        x = x.add(positionEmbed.lookup(positions));
        if (segmentEmbed != null) {
            var zeros = [for (_ in 0...seq) 0];
            x = x.add(segmentEmbed.lookup(zeros));
        }
        if (embedNorm != null) x = embedNorm.forward(x);

        for (block in blocks) x = block.forward(x);
        if (encoderNorm != null) x = encoderNorm.forward(x);
        return x;
    }

    /**
     * Module-interface forward. Accepts a 1-D Int tensor of token IDs
     * and returns hidden states `[seq, hidden]`. Production code should
     * prefer `encode()` to skip the round-trip through `Tensor`.
     */
    public function forward(x:Tensor):Tensor {
        var n = x.numel();
        var ids = [];
        for (i in 0...n) ids.push(Std.int(x.get([i])));
        return encode(ids);
    }

    public function parameters():Array<NamedTensor> {
        var ps:Array<NamedTensor> = [];
        for (p in tokenEmbed.parameters()) ps.push({ name: "token_embd." + p.name, tensor: p.tensor });
        for (p in positionEmbed.parameters()) ps.push({ name: "position_embd." + p.name, tensor: p.tensor });
        if (segmentEmbed != null) {
            for (p in segmentEmbed.parameters()) ps.push({ name: "token_types." + p.name, tensor: p.tensor });
        }
        if (embedNorm != null) {
            for (p in embedNorm.parameters()) ps.push({ name: "token_embd_norm." + p.name, tensor: p.tensor });
        }
        for (block in blocks) {
            for (p in block.parameters()) ps.push(p);
        }
        if (encoderNorm != null) {
            for (p in encoderNorm.parameters()) ps.push({ name: "output_norm." + p.name, tensor: p.tensor });
        }
        return ps;
    }
}
