package nue.arch;

import nue.EncoderModel;
import nue.Embedding;
import nue.transformer.LayerNorm;
import nue.transformer.TransformerBlock;
import nue.model.ModelMetadata;
import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.ds.BertGraph;
import rayzor.Bytes;

/**
 * BERT-family encoder. Composes the input embedding stack (token +
 * position + optional segment, with a LayerNorm on the sum), N
 * pre-norm `TransformerBlock`s, and a final encoder LayerNorm.
 *
 * The forward pass takes token IDs and returns per-token hidden
 * states `[seq_len, hidden_size]`. An optional `attentionMask` gates a
 * padded/batched sequence: padded keys get an additive large-negative
 * bias before softmax and are excluded from the mean pool. Task heads
 * (MLM, classification) live in caller code or future `nue.head.*`.
 *
 * Segment embeddings are optional — many BERT GGUFs omit them or set
 * them to zero for the single-segment case. When the loader doesn't
 * register `token_types.weight`, `segmentEmbed` is `null` and the
 * forward path skips the segment contribution.
 *
 * The encoder forward path is wired end-to-end against the `LayerNorm`
 * + `MultiHeadAttention` + `LayerNorm` + `GeluFFN` stack, with optional
 * padding-aware attention masking for batched inputs.
 */
class BertModel implements EncoderModel {
    // Optional pieces are non-null placeholders gated by a Bool flag, not
    // `Null<T>` fields (which are not reliable here across module boundaries).
    public var meta:ModelMetadata;
    /** BNNSGraph fused-encoder engine (Phase 4): the model HANDLE returned by
        `BertGraph.load`, set by the embedder. 0 = engine off. When set,
        `encode` routes the whole block stack through one graph call for
        bucket-fitting sequences. */
    public var graphHandle:Int = 0;
    public var tokenEmbed:Embedding;
    public var positionEmbed:Embedding;
    public var segmentEmbed:Embedding;   // placeholder when hasSegment == false
    public var embedNorm:LayerNorm;      // BERT always has the post-embedding norm
    public var blocks:Array<TransformerBlock>;
    public var encoderNorm:LayerNorm;    // placeholder when hasEncoderNorm == false
    public var hasSegment:Bool;
    public var hasEncoderNorm:Bool;

    // All params required and non-null; presence of the optional pieces is
    // carried by the trailing Bool flags.
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

        // Tensors are manually freed (Tensor is @:derive([Clone]), not Drop);
        // every embedding-stack intermediate below leaked one allocation per
        // encode. Bind + free each at last use.
        var h0 = tokenEmbed.lookup(tokenIds);
        var posLookup = positionEmbed.lookup(positions);
        var h1 = h0.add(posLookup);
        h0.free();
        posLookup.free();
        // token_type embedding: single-segment BERT uses segment id 0 for
        // every position, so add token_types[0] to each row.
        var h2:Tensor;
        if (hasSegment) {
            var segIds = [for (i in 0...seq) 0];
            var segLookup = segmentEmbed.lookup(segIds);
            h2 = h1.add(segLookup);
            h1.free();
            segLookup.free();
        } else {
            h2 = h1;
        }
        // Post-embedding LayerNorm — mandatory for BERT.
        var h3 = embedNorm.forward(h2);
        h2.free();

        // A mask with any zero entry (a padded/batched sequence) takes the
        // masked block path: an additive `[seq]` key bias — 0 for real tokens,
        // large-negative for padding — built once and reused across layers.
        // An unpadded single sequence (all-ones or no mask) stays maskless.
        var masked = false;
        if (attentionMask != null) {
            for (t in 0...seq) {
                if (attentionMask[t] == 0) { masked = true; break; }
            }
        }

        var hBlocks = h3;
        // Fused-graph engine: one BNNSGraph call replaces the whole block
        // loop when the sequence was padded to a loaded bucket (embedText
        // does that when the engine is on). Bias is the additive key mask in
        // fp16-safe range (-1e4, NOT the f32 path's -1e30). Falls through to
        // the per-block path on any failure.
        var ranGraph = false;
        if (graphHandle > 0 && attentionMask != null && BertGraph.bucketFor(graphHandle, seq) == seq) {
            var biasB = Bytes.alloc(seq * 4);
            for (t in 0...seq) biasB.setFloat(t * 4, attentionMask[t] == 0 ? -1e4 : 0.0);
            var gOut = Tensor.zeros([seq, meta.hiddenSize], F32);
            var rc = BertGraph.execute(graphHandle, seq, h3.data().raw(), biasB.address(), gOut.data().raw());
            biasB.free();
            if (rc == 0) {
                h3.free();
                hBlocks = gOut;
                ranGraph = true;
            } else {
                gOut.free();
            }
        }
        if (ranGraph) {
            // encoder norm / ownership handling below is shared
        } else if (masked) {
            var bias = Tensor.fromArray([for (t in 0...seq) (attentionMask[t] == 0 ? -1e30 : 0.0)], F32);
            for (block in blocks) {
                var nextH = hBlocks.clone();
                var prev = hBlocks;
                hBlocks = block.forwardMasked(nextH, bias);
                prev.free(); // old hidden state (h3 on the first pass) — was leaked per layer
            }
            bias.free();
        } else {
            for (block in blocks) {
                var nextH = hBlocks.clone();
                var prev = hBlocks;
                hBlocks = block.forward(nextH);
                prev.free();
            }
        }
        var hFinal:Tensor;
        if (hasEncoderNorm) {
            var hc = hBlocks.clone();
            hFinal = encoderNorm.forward(hc);
            hc.free();
            hBlocks.free();
        } else {
            hFinal = hBlocks; // transfer ownership to the caller — no clone/leak
        }
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
