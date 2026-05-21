package nue.arch;

import nue.Module;
import nue.Linear;
import nue.Embedding;
import nue.EncoderModel;
import nue.BertModel;
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

    public function validate(meta:ModelMetadata):Bool {
        if (meta.numHeads <= 0 || meta.numKvHeads != meta.numHeads) return false;
        if (meta.headDim * meta.numHeads != meta.hiddenSize) return false;
        if (meta.numLayers <= 0 || meta.vocabSize <= 0) return false;
        return true;
    }

    public function build(meta:ModelMetadata, weights:NamedTensorMap):Module {
        var tokenEmbed = new Embedding(
            takeWeight(weights, "token_embd.weight"),
            meta.vocabSize, meta.hiddenSize, "weight"
        );
        var positionEmbed = new Embedding(
            takeWeight(weights, "position_embd.weight"),
            meta.maxSeqLen, meta.hiddenSize, "weight"
        );
        var segmentEmbed:Embedding = null;
        if (weights.exists("token_types.weight")) {
            // Most BERTs use 2 segment IDs (sentence-A vs sentence-B).
            // The exact vocab size doesn't matter for `lookup` here.
            segmentEmbed = new Embedding(
                takeWeight(weights, "token_types.weight"),
                2, meta.hiddenSize, "weight"
            );
        }
        var embedNorm = takeOptionalNorm(weights, "token_embd_norm", meta.normEps);

        var blocks:Array<TransformerBlock> = [];
        for (i in 0...meta.numLayers) {
            blocks.push(buildBlock(meta, i, weights));
        }

        var encoderNorm = takeOptionalNorm(weights, "output_norm", meta.normEps);

        return new BertModel(
            meta, tokenEmbed, positionEmbed,
            segmentEmbed, embedNorm,
            blocks, encoderNorm
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

        var attnNorm = takeNorm(weights, prefix + "attn_norm", meta.normEps);

        var ffnUp = makeLinear(weights, prefix + "ffn_up");
        var ffnDown = makeLinear(weights, prefix + "ffn_down");
        var ffn = new GeluFFN(ffnUp, ffnDown);

        var ffnNorm = takeNorm(weights, prefix + "ffn_norm", meta.normEps);

        return new TransformerBlock(attnNorm, attn, ffnNorm, ffn, prefix);
    }

    static function makeLinear(weights:NamedTensorMap, baseName:String):Linear {
        var weight = takeWeight(weights, baseName + ".weight");
        var bias:Tensor = weights.exists(baseName + ".bias")
            ? weights.get(baseName + ".bias")
            : null;
        return new Linear(weight, bias, "weight");
    }

    static function takeNorm(weights:NamedTensorMap, baseName:String, eps:Float):LayerNorm {
        var weight = takeWeight(weights, baseName + ".weight");
        var bias:Tensor = weights.exists(baseName + ".bias")
            ? weights.get(baseName + ".bias")
            : null;
        return new LayerNorm(weight, bias, eps, "weight");
    }

    static function takeOptionalNorm(weights:NamedTensorMap, baseName:String, eps:Float):LayerNorm {
        if (!weights.exists(baseName + ".weight")) return null;
        return takeNorm(weights, baseName, eps);
    }

    static function takeWeight(weights:NamedTensorMap, name:String):Tensor {
        var t = weights.get(name);
        if (t == null) throw "nue.arch.Bert: missing weight '" + name + "'";
        return t;
    }
}
