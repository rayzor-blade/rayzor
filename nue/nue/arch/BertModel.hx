package nue.arch;

import nue.EncoderModel;
import nue.Embedding;
import nue.transformer.LayerNorm;
import nue.transformer.TransformerBlock;
import nue.model.ModelMetadata;
import rayzor.ds.Tensor;
import rayzor.ds.DType;

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
    // NO `Null<T>` instance fields: a Null<class> field reads as a corrupt
    // pointer when the object is built cross-module (BertArch) and its
    // methods run here — segmentEmbed/embedNorm crashed on access while the
    // plain-typed tokenEmbed/positionEmbed/blocks fields were fine (see
    // bugs_null_class_field_xmodule). Optional pieces are stored as a
    // non-null placeholder gated by a Bool flag instead.
    public var meta:ModelMetadata;
    public var tokenEmbed:Embedding;
    public var positionEmbed:Embedding;
    public var segmentEmbed:Embedding;   // placeholder when hasSegment == false
    public var embedNorm:LayerNorm;      // BERT always has the post-embedding norm
    public var blocks:Array<TransformerBlock>;
    public var encoderNorm:LayerNorm;    // placeholder when hasEncoderNorm == false
    public var hasSegment:Bool;
    public var hasEncoderNorm:Bool;

    // All params required and non-null. Optional params in the middle of the
    // signature ALSO misbind fields cross-module, so presence is carried by
    // the trailing Bool flags. See bugs_import_xmodule_member_resolution.
    public function new(
        meta:ModelMetadata,
        tokenEmbed:Embedding,
        positionEmbed:Embedding,
        segmentEmbed:Embedding,
        embedNorm:LayerNorm,
        blocks:Array<TransformerBlock>,
        encoderNorm:LayerNorm,
        hasSegment:Bool,
        hasEncoderNorm:Bool
    ) {
        this.meta = meta;
        this.tokenEmbed = tokenEmbed;
        this.positionEmbed = positionEmbed;
        this.segmentEmbed = segmentEmbed;
        this.embedNorm = embedNorm;
        this.blocks = blocks;
        this.encoderNorm = encoderNorm;
        this.hasSegment = hasSegment;
        this.hasEncoderNorm = hasEncoderNorm;
    }

    public function encode(tokenIds:Array<Int>, ?attentionMask:Array<Int>):Tensor {
        var seq = tokenIds.length;
        var positions = [for (i in 0...seq) i];

        var h0 = tokenEmbed.lookup(tokenIds);
        var h1 = h0.add(positionEmbed.lookup(positions));
        // token_type embedding: single-segment BERT uses segment id 0 for
        // every position, so add token_types[0] to each row.
        var h2:Tensor;
        if (hasSegment) {
            var segIds = [for (i in 0...seq) 0];
            h2 = h1.add(segmentEmbed.lookup(segIds));
        } else {
            h2 = h1;
        }
        // Post-embedding LayerNorm — mandatory for BERT.
        var h3 = embedNorm.forward(h2);

        var hBlocks = h3;
        for (block in blocks) {
            var nextH = hBlocks.clone();
            hBlocks = block.forward(nextH);
        }
        var hFinal:Tensor = hasEncoderNorm
            ? encoderNorm.forward(hBlocks.clone())
            : hBlocks.clone();
        return hFinal;
    }

    /**
     * Sentence embedding: encode, then **mean-pool over tokens weighted
     * by the attention mask, then L2-normalize** — the sentence-
     * transformers convention for all-MiniLM / bge / e5 (pooling_type =
     * MEAN). Returns a 1-D `[hidden]` unit vector.
     *
     * The pool runs on an Int denominator and a plain Float accumulator
     * array (not a loop-carried conditional Float) to avoid the
     * per-element boxing path; it's O(seq*hidden), negligible beside the
     * block matmuls.
     */
    public function embed(tokenIds:Array<Int>, ?attentionMask:Array<Int>):Tensor {
        var hidden = encode(tokenIds, attentionMask); // [seq, hidden]
        var hs = meta.hiddenSize;
        var seq = tokenIds.length;

        var acc = [for (_ in 0...hs) 0.0];
        var counted = 0;
        for (t in 0...seq) {
            var m = (attentionMask != null) ? attentionMask[t] : 1;
            if (m == 0) continue;
            counted++;
            var base = t * hs;
            for (j in 0...hs) acc[j] += hidden.getFlat(base + j);
        }
        hidden.free();

        var denom = counted > 0 ? counted : 1;
        var ss = 0.0;
        for (j in 0...hs) {
            var v = acc[j] / denom;
            acc[j] = v;
            ss += v * v;
        }
        var inv = 1.0 / Math.sqrt(ss);
        for (j in 0...hs) acc[j] = acc[j] * inv;

        return Tensor.fromArray(acc, F32);
    }

    /**
     * Module-interface forward. Accepts a 1-D Int tensor of token IDs
     * and returns hidden states `[seq, hidden]`. Production code should
     * prefer `encode()` to skip the round-trip through `Tensor`.
     */
    public function forward(x:Tensor):Tensor {
        var n = x.numel();
        var ids = [];
        for (i in 0...n) ids.push(Std.int(x.getFlat(i)));
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
