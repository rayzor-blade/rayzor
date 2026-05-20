package nue.arch;

import nue.Module;
import nue.Linear;
import nue.Embedding;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import nue.transformer.RMSNorm;
import nue.transformer.RoPE;
import nue.transformer.SwiGLU;
import nue.transformer.KVCache;
import nue.transformer.GQAttention;
import nue.transformer.TransformerBlock;
import rayzor.ds.Tensor;
import rayzor.ds.DType;

/**
 * Llama-family architecture builder. Covers Llama 1/2/3, Mistral,
 * Qwen2 — every causal LM that follows the
 *   `pre-RMSNorm → GQAttention(+RoPE) → residual → pre-RMSNorm → SwiGLU → residual`
 * recipe with no learned positional bias.
 *
 * Reads weights from the GGUF naming convention:
 * ```
 *   token_embd.weight                       — input embedding
 *   blk.{L}.attn_norm.weight                — pre-attention RMSNorm gain
 *   blk.{L}.attn_q.weight                   — Q projection
 *   blk.{L}.attn_k.weight                   — K projection
 *   blk.{L}.attn_v.weight                   — V projection
 *   blk.{L}.attn_output.weight              — output projection
 *   blk.{L}.ffn_norm.weight                 — pre-FFN RMSNorm gain
 *   blk.{L}.ffn_gate.weight                 — SwiGLU gate
 *   blk.{L}.ffn_up.weight                   — SwiGLU up
 *   blk.{L}.ffn_down.weight                 — SwiGLU down
 *   output_norm.weight                      — final RMSNorm gain
 *   output.weight (optional, when not tied) — LM head
 * ```
 * Other file formats (ONNX, safetensors) should normalise their tensor
 * names to this scheme in the loader; the architecture builder then
 * works against a single convention.
 */
class LlamaArch implements ArchBuilder {
    public function new() {}

    public function name():String {
        return "llama";
    }

    public function validate(meta:ModelMetadata):Bool {
        if (meta.numHeads <= 0 || meta.numKvHeads <= 0) return false;
        if (meta.numHeads % meta.numKvHeads != 0) return false;
        if (meta.headDim * meta.numHeads != meta.hiddenSize) return false;
        if (meta.numLayers <= 0 || meta.vocabSize <= 0) return false;
        return true;
    }

    public function build(meta:ModelMetadata, weights:NamedTensorMap):Module {
        var dtype = inferDType(weights, "token_embd.weight");
        var rope = new RoPE(meta.headDim, meta.maxSeqLen, meta.ropeBase);

        var embed = new Embedding(
            takeWeight(weights, "token_embd.weight"),
            meta.vocabSize, meta.hiddenSize, "weight"
        );

        var blocks:Array<Module> = [];
        for (i in 0...meta.numLayers) {
            blocks.push(buildBlock(meta, i, dtype, rope, weights));
        }

        var outputNorm = new RMSNorm(
            takeWeight(weights, "output_norm.weight"),
            meta.normEps, "weight"
        );

        // LM head — tied with embed weight when the file says so.
        var lmHead = if (meta.tieWordEmbeddings)
            new Linear(embed.embedTable(), null, "weight")
        else
            new Linear(takeWeight(weights, "output.weight"), null, "weight");

        return new LlamaModel(meta, embed, blocks, outputNorm, lmHead, rope);
    }

    static function buildBlock(
        meta:ModelMetadata, layerIndex:Int, dtype:DType,
        rope:RoPE, weights:NamedTensorMap
    ):TransformerBlock {
        var prefix = "blk." + layerIndex + ".";
        var attnNorm = new RMSNorm(
            takeWeight(weights, prefix + "attn_norm.weight"),
            meta.normEps, "weight"
        );

        var cache = new KVCache(meta.maxSeqLen, meta.numKvHeads, meta.headDim, dtype);
        var qProj = new Linear(takeWeight(weights, prefix + "attn_q.weight"), null, "weight");
        var kProj = new Linear(takeWeight(weights, prefix + "attn_k.weight"), null, "weight");
        var vProj = new Linear(takeWeight(weights, prefix + "attn_v.weight"), null, "weight");
        var oProj = new Linear(takeWeight(weights, prefix + "attn_output.weight"), null, "weight");
        var attn = new GQAttention(
            qProj, kProj, vProj, oProj, rope, cache,
            meta.numHeads, meta.numKvHeads, meta.headDim
        );

        var ffnNorm = new RMSNorm(
            takeWeight(weights, prefix + "ffn_norm.weight"),
            meta.normEps, "weight"
        );

        var ffn = new SwiGLU(
            new Linear(takeWeight(weights, prefix + "ffn_gate.weight"), null, "weight"),
            new Linear(takeWeight(weights, prefix + "ffn_up.weight"), null, "weight"),
            new Linear(takeWeight(weights, prefix + "ffn_down.weight"), null, "weight")
        );

        return new TransformerBlock(attnNorm, attn, ffnNorm, ffn, prefix);
    }

    /**
     * Pull a tensor out of the weights map by name. Throws (rather than
     * silently zero-initing) when the file is missing a weight the
     * architecture needs — surfaces bad checkpoints early.
     */
    static function takeWeight(weights:NamedTensorMap, name:String):Tensor {
        var t = weights.get(name);
        if (t == null) throw "nue.arch.Llama: missing weight '" + name + "'";
        return t;
    }

    static function inferDType(weights:NamedTensorMap, sentinel:String):DType {
        var t = weights.get(sentinel);
        if (t == null) return F32;
        return t.dtype();
    }
}
