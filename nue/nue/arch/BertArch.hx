package nue.arch;

import nue.Module;
import nue.Linear;
import nue.Embedding;
import nue.EncoderModel;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import nue.transformer.LayerNorm;
import nue.transformer.MultiHeadAttention;
import nue.transformer.GeluFFN;
import nue.transformer.TransformerBlock;
import rayzor.ds.Tensor;

/**
 * Bidirectional encoder builder — BERT / RoBERTa / DeBERTa family.
 *
 * The encoder composes:
 *   - token embedding + segment embedding (optional) + absolute
 *     position embedding, summed before block 0
 *   - optional LayerNorm on the embedding sum
 *   - N × TransformerBlock(LayerNorm, MultiHeadAttention, LayerNorm, GeluFFN)
 *   - optional final LayerNorm
 *   - task heads (MLM, classification) live in downstream user code
 *     or future `nue.head.*` modules
 *
 * BERT weight naming differs from PyTorch / HF defaults; the loader
 * is responsible for normalising into the GGUF canonical scheme this
 * builder reads against. Names this builder expects:
 * ```
 *   token_embd.weight                       — input embedding
 *   position_embd.weight                    — learned absolute positions
 *   token_types.weight (optional)           — segment embedding
 *   token_embd_norm.weight (+ .bias)        — pre-block LayerNorm
 *   blk.{L}.attn_q.weight (+ .bias)         — Q projection
 *   blk.{L}.attn_k.weight (+ .bias)         — K projection
 *   blk.{L}.attn_v.weight (+ .bias)         — V projection
 *   blk.{L}.attn_output.weight (+ .bias)    — output projection
 *   blk.{L}.attn_norm.weight (+ .bias)      — post-attention LayerNorm
 *   blk.{L}.ffn_up.weight (+ .bias)         — FFN expansion
 *   blk.{L}.ffn_down.weight (+ .bias)       — FFN projection
 *   blk.{L}.ffn_norm.weight (+ .bias)       — post-FFN LayerNorm
 *   output_norm.weight (+ .bias) (optional) — final encoder LayerNorm
 * ```
 */
class BertArch implements ArchBuilder {
    public function new() {}

    public function name():String {
        return "bert";
    }

    /**
     * `NUE_BERT_QUANT=1` keeps encoder weights in their GGUF quantised form
     * (`QTensor`) instead of dequantising to F32 at load. Off by default,
     * on accuracy: measured on all-MiniLM-L6-v2-Q8_0 against the F32 goldens,
     * the native path scores min/mean cosine 0.9934 / 0.9970 versus
     * 0.99969 / 0.99985 dequantised — under this repo's >= 0.999 gate. The
     * loss is activation quantisation (int8 × int8), not weight precision,
     * so it is inherent to the kernel rather than a defect. Opt in when
     * encoder memory matters more than the last fraction of a percent of
     * cosine; generative models keep quantised weights unconditionally
     * because sampling tolerates it and the memory win is far larger.
     *
     * Cached like `Linear.useHaxeMatmul` (0 = uninitialised — load-bearing
     * for cross-module static duplication).
     */
    static var _quantWeights:Int = 0;

    public static function useQuantWeights():Bool {
        if (_quantWeights == 0) {
            var v = Sys.getEnvOr("NUE_BERT_QUANT", "RAYZOR_BERT_QUANT");
            _quantWeights = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _quantWeights == 1;
    }

    public function validate(meta:ModelMetadata):Bool {
        if (meta.numHeads <= 0 || meta.numKvHeads != meta.numHeads) return false;
        if (meta.headDim * meta.numHeads != meta.hiddenSize) return false;
        if (meta.numLayers <= 0 || meta.vocabSize <= 0) return false;
        return true;
    }

    public function build(meta:ModelMetadata, weights:NamedTensorMap):Module {
        // Encoder GGUFs commonly ship `token_embd` quantised (the shipped
        // MiniLM Q8_0 does), so a QTensor is the normal arrival shape here —
        // never demand an F32 table. `useQuantWeights` picks the
        // representation; see `makeLinear` for the same split.
        var tokenEmbed:Embedding;
        if (weights.isQuantised("token_embd.weight")) {
            var qEmbed = weights.getQuant("token_embd.weight");
            if (qEmbed == null) {
                throw "nue.arch.Bert: missing quant weight 'token_embd.weight'";
            }
            if (useQuantWeights()) {
                tokenEmbed = Embedding.fromQuant(qEmbed, meta.vocabSize, meta.hiddenSize, "weight");
            } else {
                tokenEmbed = new Embedding(qEmbed.dequant(), meta.vocabSize, meta.hiddenSize, "weight");
            }
        } else {
            tokenEmbed = new Embedding(
                takeWeight(weights, "token_embd.weight"),
                meta.vocabSize, meta.hiddenSize, "weight"
            );
        }
        var positionEmbed = new Embedding(
            takeWeight(weights, "position_embd.weight"),
            meta.maxSeqLen, meta.hiddenSize, "weight"
        );

        // Post-embedding LayerNorm is REQUIRED for BERT (embeddings.LayerNorm
        // in HF, token_embd_norm in GGUF) — a missing one silently zeroes the
        // encoder's input scale, so fail loudly instead of skipping it.
        var embedNorm = takeNormAny(weights, ["token_embd_norm"], meta.normEps);

        // BertModel stores optional pieces as non-null placeholders + a Bool
        // presence flag; pass a harmless placeholder when the weight is absent.
        var hasSegment = weights.exists("token_types.weight");
        var segmentEmbed = hasSegment
            ? new Embedding(takeWeight(weights, "token_types.weight"), 2, meta.hiddenSize, "weight")
            : tokenEmbed; // placeholder, never looked up when hasSegment == false

        var blocks:Array<TransformerBlock> = [];
        for (i in 0...meta.numLayers) {
            blocks.push(buildBlock(meta, i, weights));
        }

        var hasEncoderNorm = weights.exists("output_norm.weight");
        var encoderNorm = hasEncoderNorm
            ? takeNorm(weights, "output_norm", meta.normEps)
            : embedNorm; // placeholder, never applied when hasEncoderNorm == false

        return new BertModel(
            meta, tokenEmbed, positionEmbed,
            segmentEmbed, embedNorm,
            blocks, encoderNorm,
            hasSegment, hasEncoderNorm
        );
    }

    static function buildBlock(
        meta:ModelMetadata, layerIndex:Int, weights:NamedTensorMap
    ):TransformerBlock {
        var prefix = "blk." + layerIndex + ".";

        var qProj = makeLinear(weights, prefix + "attn_q");
        var kProj = makeLinear(weights, prefix + "attn_k");
        var vProj = makeLinear(weights, prefix + "attn_v");
        var oProj = makeLinear(weights, prefix + "attn_output");
        var attn = new MultiHeadAttention(qProj, kProj, vProj, oProj, meta.numHeads, meta.headDim);

        // Post-attention LayerNorm. GGUF names it
        // `attn_output_norm`; the SafetensorsLoader normalizes to
        // `attn_norm`. Accept either so both load paths resolve.
        var attnNorm = takeNormAny(weights, [prefix + "attn_norm", prefix + "attn_output_norm"], meta.normEps);

        var ffnUp = makeLinear(weights, prefix + "ffn_up");
        var ffnDown = makeLinear(weights, prefix + "ffn_down");
        var ffn = new GeluFFN(ffnUp, ffnDown);

        // Post-FFN LayerNorm. GGUF names it `layer_output_norm`;
        // SafetensorsLoader normalizes to `ffn_norm`.
        var ffnNorm = takeNormAny(weights, [prefix + "ffn_norm", prefix + "layer_output_norm"], meta.normEps);

        // BERT is POST-norm: norms apply to the residual sum, not the
        // sublayer input. `attn_output_norm` / `layer_output_norm` are
        // trained for that position — pre-norm diverges every layer.
        return new TransformerBlock(attnNorm, attn, ffnNorm, ffn, prefix, true);
    }

    static function makeLinear(weights:NamedTensorMap, baseName:String):Linear {
        // Biases stay F32 (GGUF keeps them unquantised); only the weight can
        // arrive as a QTensor.
        var bias:Tensor = weights.exists(baseName + ".bias")
            ? weights.get(baseName + ".bias")
            : null;
        if (weights.isQuantised(baseName + ".weight")) {
            var qw = weights.getQuant(baseName + ".weight");
            if (qw == null) {
                throw "nue.arch.Bert: missing quant weight '" + baseName + ".weight'";
            }
            if (useQuantWeights()) return Linear.fromQuant(qw, bias, "weight");
            return new Linear(qw.dequant(), bias, "weight");
        }
        return new Linear(takeWeight(weights, baseName + ".weight"), bias, "weight");
    }

    static function takeNorm(weights:NamedTensorMap, baseName:String, eps:Float):LayerNorm {
        var weight = takeWeight(weights, baseName + ".weight");
        var bias:Tensor = weights.exists(baseName + ".bias")
            ? weights.get(baseName + ".bias")
            : null;
        return new LayerNorm(weight, bias, eps, "weight");
    }

    /**
     * Resolve a LayerNorm from the first candidate base-name that exists.
     * BERT norms carry different names across loaders — GGUF
     * uses `attn_output_norm` / `layer_output_norm`, the SafetensorsLoader
     * normalizes to `attn_norm` / `ffn_norm` — so both must resolve. Throws
     * if none match (these norms are mandatory for a correct encoder).
     */
    static function takeNormAny(weights:NamedTensorMap, baseNames:Array<String>, eps:Float):LayerNorm {
        for (b in baseNames) {
            if (weights.exists(b + ".weight")) return takeNorm(weights, b, eps);
        }
        throw "nue.arch.Bert: missing required LayerNorm, tried " + baseNames.join(", ");
    }

    static function takeWeight(weights:NamedTensorMap, name:String):Tensor {
        // Check `isQuantised` FIRST. A quantised weight lives only in the
        // QTensor map, so `get` would miss it — and `StringMap.get` on a
        // missing key returns a raw 0 rather than null (bugs_stringmap_null_get),
        // so the null guard below cannot be trusted to catch it. Without this
        // the caller receives a 0-handle Tensor and dies later inside a kernel.
        if (weights.isQuantised(name)) {
            throw "nue.arch.Bert: weight '" + name + "' is quantised; "
                + "this call site needs the QTensor path";
        }
        var t = weights.get(name);
        if (t == null) throw "nue.arch.Bert: missing weight '" + name + "'";
        return t;
    }
}
