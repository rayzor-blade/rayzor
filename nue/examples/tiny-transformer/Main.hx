import nue.Module;
import nue.Embedding;
import nue.Linear;
import nue.arch.LlamaModel;
import nue.model.ModelMetadata;
import nue.transformer.RMSNorm;
import nue.transformer.RoPE;
import nue.transformer.KVCache;
import nue.transformer.GQAttention;
import nue.transformer.SwiGLU;
import nue.transformer.TransformerBlock;
import nue.sampling.ArgmaxSampler;
import nue.sampling.GenerationLoop;
import nue.tokenizer.Vocab;
import nue.tokenizer.BPETokenizer;
import rayzor.ds.Tensor;
import rayzor.ds.DType;

/**
 * Synthetic tiny Llama-style smoke test.
 *
 * Builds a 2-layer / 16-hidden / 2-Q-head / 1-KV-head model with
 * deterministic weights and runs both a one-shot forward pass and a
 * 5-token argmax generation. The goal is wiring + shape correctness —
 * not coherent output — so any crash, NaN, or shape mismatch is
 * immediately visible without needing a real GGUF download.
 */
class Main {
    static inline var VOCAB = 8;
    static inline var HIDDEN = 16;
    static inline var LAYERS = 2;
    static inline var NHEADS = 2;
    static inline var NKVH = 1;
    static inline var HEAD_DIM = 8;
    static inline var FFN = 32;
    static inline var MAX_SEQ = 16;

    static function main() {
        trace("=== nue tiny-transformer smoke test ===");

        var meta:ModelMetadata = {
            architecture: "llama",
            hiddenSize: HIDDEN,
            intermediateSize: FFN,
            numLayers: LAYERS,
            numHeads: NHEADS,
            numKvHeads: NKVH,
            headDim: HEAD_DIM,
            vocabSize: VOCAB,
            maxSeqLen: MAX_SEQ,
            normEps: 1e-5,
            ropeBase: 10000.0,
            tieWordEmbeddings: false
        };

        var rope = new RoPE(HEAD_DIM, MAX_SEQ, meta.ropeBase);
        var embed = new Embedding(
            deterministicWeight([VOCAB, HIDDEN], 0.01, 1),
            VOCAB, HIDDEN, "weight"
        );
        var blocks:Array<Module> = [];
        for (i in 0...LAYERS) blocks.push(buildBlock(i));
        var outputNorm = new RMSNorm(onesWeight(HIDDEN), meta.normEps, "weight");
        var lmHead = new Linear(deterministicWeight([HIDDEN, VOCAB], 0.02, 9));
        var model = new LlamaModel(meta, embed, blocks, outputNorm, lmHead, rope);

        trace("[build] LlamaModel hidden=" + HIDDEN + " layers=" + LAYERS
            + " heads=" + NHEADS + "/" + NKVH + " ffn=" + FFN);

        // Single-pass forward.
        var promptIds = [1, 2, 3];
        var logits = model.forwardIds(promptIds);
        var lshape = logits.shape();
        trace("[forward] ids=" + promptIds + " -> logits=" + describeShape(lshape));
        if (lshape[0] != promptIds.length || lshape[1] != VOCAB) {
            throw "Unexpected logits shape: " + describeShape(lshape);
        }
        model.resetCache();

        // Streaming generation with argmax.
        var vocab = new Vocab();
        for (i in 0...VOCAB) vocab.add("t" + i);
        var tok = new BPETokenizer(vocab, [], false);
        var sampler = new ArgmaxSampler();
        var loop = new GenerationLoop(model, tok, sampler, -1, 5);

        var generated:Array<Int> = [];
        loop.generate("t1", function(id:Int, _partial:String):Bool {
            generated.push(id);
            trace("[step] id=" + id);
            return true;
        });

        if (generated.length != 5) {
            throw "Expected 5 generated tokens, got " + generated.length;
        }

        // Determinism: argmax + same weights ⇒ identical sequence.
        model.resetCache();
        var generated2:Array<Int> = [];
        loop.generate("t1", function(id:Int, _p:String):Bool {
            generated2.push(id);
            return true;
        });
        for (i in 0...generated.length) {
            if (generated[i] != generated2[i]) {
                throw "Non-deterministic: step " + i
                    + " got " + generated[i] + " then " + generated2[i];
            }
        }
        trace("[check] deterministic across two runs");

        trace("=== PASS ===");
    }

    static function deterministicWeight(shape:Array<Int>, scale:Float, seed:Int):Tensor {
        var n = 1;
        for (d in shape) n *= d;
        var data:Array<Float> = [];
        for (i in 0...n) {
            var k = (seed * 31 + i * 17) % 127;
            data.push((k - 63) * scale / 63.0);
        }
        return Tensor.fromArray(data, F32).reshape(shape);
    }

    static function onesWeight(n:Int):Tensor {
        var data:Array<Float> = [];
        for (i in 0...n) data.push(1.0);
        return Tensor.fromArray(data, F32);
    }

    static function buildBlock(layerIdx:Int):TransformerBlock {
        var rope = new RoPE(HEAD_DIM, MAX_SEQ, 10000.0);
        var cache = new KVCache(MAX_SEQ, NKVH, HEAD_DIM, F32);

        var attnNorm = new RMSNorm(onesWeight(HIDDEN), 1e-5, "weight");
        var qProj = new Linear(deterministicWeight([HIDDEN, NHEADS * HEAD_DIM], 0.05, layerIdx * 7 + 1));
        var kProj = new Linear(deterministicWeight([HIDDEN, NKVH * HEAD_DIM], 0.05, layerIdx * 7 + 2));
        var vProj = new Linear(deterministicWeight([HIDDEN, NKVH * HEAD_DIM], 0.05, layerIdx * 7 + 3));
        var oProj = new Linear(deterministicWeight([NHEADS * HEAD_DIM, HIDDEN], 0.05, layerIdx * 7 + 4));
        var attn = new GQAttention(qProj, kProj, vProj, oProj, rope, cache,
            NHEADS, NKVH, HEAD_DIM);

        var ffnNorm = new RMSNorm(onesWeight(HIDDEN), 1e-5, "weight");
        var ffn = new SwiGLU(
            new Linear(deterministicWeight([HIDDEN, FFN], 0.03, layerIdx * 7 + 5)),
            new Linear(deterministicWeight([HIDDEN, FFN], 0.03, layerIdx * 7 + 6)),
            new Linear(deterministicWeight([FFN, HIDDEN], 0.03, layerIdx * 7 + 7))
        );

        return new TransformerBlock(attnNorm, attn, ffnNorm, ffn, "blk." + layerIdx + ".");
    }

    static function describeShape(s:Array<Int>):String {
        var parts:Array<String> = [];
        for (d in s) parts.push("" + d);
        return "[" + parts.join(",") + "]";
    }
}
