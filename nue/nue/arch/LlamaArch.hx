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
import rayzor.ds.QTensor;
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
        var nh = meta.numHeads;
        var nkv = meta.numKvHeads;
        if (nh <= 0 || nkv <= 0) return false;
        if (nh % nkv != 0) return false;
        var hd = meta.headDim;
        var hs = meta.hiddenSize;
        if (hd * nh != hs) return false;
        var nl = meta.numLayers;
        var vs = meta.vocabSize;
        if (nl <= 0 || vs <= 0) return false;
        return true;
    }

    public function build(meta:ModelMetadata, weights:NamedTensorMap):Module {
        var dtype = inferDType(weights, "token_embd.weight");
        var rope = new RoPE(meta.headDim, meta.maxSeqLen, meta.ropeBase);

        // Phase 4b: the embedding stays Q6_K-native — `Embedding.fromQuant`
        // routes lookups through `QTensor.gatherRowsQ6K`, dequantising only
        // the per-step token rows instead of the full `[vocab, hidden]`
        // table. Byte-identical to the prior eager-dequant + F32 gather
        // path because both share the Q6_K block decoder. Falls back to
        // F32 when the GGUF stored `token_embd.weight` as F32 (older
        // Llama-2 / Mistral checkpoints).
        var embed:Embedding;
        if (weights.isQuantised("token_embd.weight")) {
            var qEmbed = weights.getQuant("token_embd.weight");
            if (qEmbed == null) {
                throw "nue.arch.Llama: missing quant weight 'token_embd.weight'";
            }
            embed = Embedding.fromQuant(qEmbed,
                meta.vocabSize, meta.hiddenSize, "weight");
        } else {
            embed = new Embedding(takeWeight(weights, "token_embd.weight"),
                meta.vocabSize, meta.hiddenSize, "weight");
        }

        var blocks:Array<Module> = [];
        for (i in 0...meta.numLayers) {
            blocks.push(buildBlock(meta, i, dtype, rope, weights));
        }

        var outputNorm = new RMSNorm(
            takeWeight(weights, "output_norm.weight"),
            meta.normEps, "weight"
        );

        // LM head — tied with embed weight when the metadata says so OR
        // when no separate output weight exists (e.g. Llama 3.2 1B's GGUF
        // omits both the `tie_word_embeddings` flag and `output.weight`,
        // implying tied embeddings). When the embed is Q6_K-backed we
        // build the LM head via `Linear.fromQuant` so the same QTensor
        // is shared (no F32 weight copy); when the embed is F32 we wrap
        // via the standard `Linear` constructor.
        var lmHead:Linear;
        if (meta.tieWordEmbeddings || !weights.exists("output.weight")) {
            if (embed.qweight != null) {
                // Re-quantise the tied lm_head from Q6_K to Q4_K_M.
                // Q6_K SDOT (5f23311) still pays the 6-bit reconstruction
                // overhead per block in its inner loop; the Q4_K_M SDOT
                // path skips that work entirely because the 4-bit nibbles
                // already match the SDOT operand format. ~+21% wall on the
                // MZ-story 600-tok workload, on top of every prior
                // optimisation, with MATCH-on-canonical preserved.
                //
                // Embedding still uses the Q6_K source (precise lookups
                // matter for input fidelity); only the lm_head path goes
                // through the cheaper Q4_K_M view.
                //
                // Set `RAYZOR_REQUANT_LM_HEAD=0` to opt out — kept as a
                // safety hatch in case a future model exposes the naive
                // encoder's small per-block quality loss.
                var lmQt = embed.qweight;
                var optOut = Sys.getEnv("RAYZOR_REQUANT_LM_HEAD");
                if (optOut != "0") {
                    var rq = embed.qweight.requantQ6KToQ4KM();
                    if (rq != null) {
                        lmQt = rq;
                        Sys.println("[lm_head] re-quantised Q6_K → Q4_K_M for faster SDOT path");
                    }
                }
                lmHead = Linear.fromQuant(lmQt, null, "weight");
            } else {
                lmHead = new Linear(embed.weight, null, "weight");
            }
        } else {
            lmHead = buildLinear(weights, "output.weight", null);
        }

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
        var qProj = buildLinear(weights, prefix + "attn_q.weight", null);
        var kProj = buildLinear(weights, prefix + "attn_k.weight", null);
        var vProj = buildLinear(weights, prefix + "attn_v.weight", null);
        var oProj = buildLinear(weights, prefix + "attn_output.weight", null);
        var attn = new GQAttention(
            qProj, kProj, vProj, oProj, rope, cache,
            meta.numHeads, meta.numKvHeads, meta.headDim
        );

        var ffnNorm = new RMSNorm(
            takeWeight(weights, prefix + "ffn_norm.weight"),
            meta.normEps, "weight"
        );

        var ffn = new SwiGLU(
            buildLinear(weights, prefix + "ffn_gate.weight", null),
            buildLinear(weights, prefix + "ffn_up.weight", null),
            buildLinear(weights, prefix + "ffn_down.weight", null)
        );

        return new TransformerBlock(attnNorm, attn, ffnNorm, ffn, prefix);
    }

    /** Build a Linear from a weight name, picking the QTensor path when
     *  the loader registered a quantised entry. Bias is also F32 in GGUF
     *  even for Q4_K_M weights, so it's read via the F32 path. */
    static function buildLinear(weights:NamedTensorMap, weightName:String,
                                biasName:Null<String>):Linear {
        var b:Null<Tensor> = (biasName == null) ? null : weights.get(biasName);
        if (weights.isQuantised(weightName)) {
            var qw = weights.getQuant(weightName);
            if (qw == null) {
                throw "nue.arch.Llama: missing quant weight '" + weightName + "'";
            }
            return Linear.fromQuant(qw, b, "weight");
        }
        return new Linear(takeWeight(weights, weightName), b, "weight");
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

    /** Same as `takeWeight` but dequants on the fly if the entry was
     *  stored as a QTensor. Used for tensors that the F32 kernels need
     *  in materialised form (embedding lookup, RMSNorm scale). */
    static function takeWeightOrDequant(weights:NamedTensorMap, name:String):Tensor {
        if (weights.isQuantised(name)) {
            var qw = weights.getQuant(name);
            if (qw == null) {
                throw "nue.arch.Llama: missing quant weight '" + name + "'";
            }
            return qw.dequant();
        }
        return takeWeight(weights, name);
    }

    static function inferDType(weights:NamedTensorMap, sentinel:String):DType {
        var t = weights.get(sentinel);
        if (t == null) return F32;
        return t.dtype();
    }
}
