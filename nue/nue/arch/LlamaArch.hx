package nue.arch;

import nue.Module;
import nue.Linear;
import nue.Q4Matmul;
import nue.Embedding;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import nue.transformer.RMSNorm;
import nue.transformer.RoPE;
import nue.transformer.SwiGLU;
import nue.transformer.KVCache;
import nue.transformer.GQAttention;
import nue.transformer.LlamaBlock;
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
        // RoPE convention is architecture-specific. Llama/Mistral bake the
        // interleaved (NORM) convention into permuted GGUF weights; Qwen2
        // (and other NeoX-family archs) keep weights unpermuted and need the
        // half-split (NEOX) rotation. Wrong choice → degenerate attention.
        rope.neox = (meta.architecture == "qwen2");

        // Q8_0 KV cache opt-in. Default off until we have wider
        // workload coverage; `RAYZOR_KV_Q8=1` enables. Storage drops
        // from `[max_seq, kv_heads, head_dim, F32]` to ~3.76× smaller
        // Q8_0 blocks; decode dispatches through the fused
        // `flashAttnDecodeQ8` kernel (single-pass dequant-on-load).
        // Read here (in a method, threaded through ctor) rather than
        // inside KVCache.new so the per-layer construction is free of
        // env-lookup overhead and the build chain stays self-contained.
        // DEFAULT ON (opt out with `NUE_KV_Q8=0`). Every shipped runner set it,
        // so leaving it opt-in meant a plain `rayzor run` carried an F32 KV
        // cache: measured on Mistral-7B, KV_Q8+FLASH is worth -1.02 GB RSS
        // (5.54 -> 4.52) on top of the Haxe matmul, with BIT-IDENTICAL output
        // across a 60-token Qwen generation.
        var kvQ8Env = Sys.getEnvOr("NUE_KV_Q8", "RAYZOR_KV_Q8");
        var useKvQ8 = !(kvQ8Env != null && (kvQ8Env == "0" || kvQ8Env == "false"));
        // Guest-owned Q8 cache + pure-Haxe flash decode (see FlashDecode).
        // Read here, alongside RAYZOR_KV_Q8, and threaded through as an
        // explicit parameter: this build path is a proven-safe env-read
        // site, and an explicit param avoids gate resolution inside the
        // KVCache ctor.
        var haxeMatmul = Linear.useHaxeMatmul();
        var flashEnv = Sys.getEnvOr("NUE_FLASH", "RAYZOR_HAXE_FLASH");
        // NOT gated on `haxeMatmul`. The Q8 KV cache + Haxe flash decode are a
        // separate subsystem — KVCache takes no SpinPool and never calls the
        // matmul kernel — so conjoining them made `NUE_MATMUL` swap three
        // subsystems at once and no Haxe-vs-Rust matmul number was ever a
        // matmul comparison. `NUE_MATMUL` selects the kernel; `NUE_FLASH` and
        // `NUE_KV_Q8` select attention. One flag, one variable.
        // Flash decode is also DEFAULT ON (opt out with `NUE_FLASH=0`) — it is
        // the pure-Haxe attention path, which is the shipping direction.
        var useHaxeQ8 = useKvQ8
            && !(flashEnv != null && (flashEnv == "0" || flashEnv == "false"));

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

        var blocks:Array<LlamaBlock> = [];
        // One persistent spin pool shared by every quant Linear (pure-Haxe
        // matmul path only). Owned by the model; Main must shut it down.
        var sp:Null<rayzor.concurrent.SpinPool> = null;
        if (haxeMatmul) sp = new rayzor.concurrent.SpinPool(Q4Matmul.workerCount(), Q4Matmul.poolSpins(), Q4Matmul.poolRelax(), Q4Matmul.poolProfiling());
        // Spin policy is caller-owned; NUE_POOL_ADAPTIVE=0 restores the
        // iteration-only tight-spin bound for A/B against the stall.
        if (sp != null) sp.adaptiveTight = Q4Matmul.poolAdaptive();

        for (i in 0...meta.numLayers) {
            blocks.push(buildBlock(meta, i, dtype, rope, weights, useKvQ8, useHaxeQ8, sp));
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
                var optOut = Sys.getEnvOr("NUE_REQUANT_LM_HEAD", "RAYZOR_REQUANT_LM_HEAD");
                if (optOut != "0") {
                    var rq = embed.qweight.requantQ6KToQ4KM();
                    if (rq != null) {
                        lmQt = rq;
                        Sys.println("[lm_head] re-quantised Q6_K → Q4_K_M for faster SDOT path");
                    }
                }
                lmHead = Linear.fromQuant(lmQt, null, "weight");
                lmHead.pool = sp;
            } else {
                lmHead = new Linear(embed.weight, null, "weight");
            }
        } else {
            lmHead = buildLinear(weights, "output.weight", null, sp);
        }

        var model = new LlamaModel(meta, embed, blocks, outputNorm, lmHead, rope);
        model.spinPool = sp;

        // Write down what was just built, under the flag that already asks for
        // the kernel census. Everything recorded is a value this function
        // already holds or can read back off the model, so describing the build
        // cannot change it — in particular no fusion gate is consulted here,
        // which would move the census's own first-read line.
        var censusEnv = Sys.getEnvOr("NUE_DUMP_Q4_GATES", "RAYZOR_DUMP_Q4_GATES");
        if (censusEnv != null && censusEnv != "0" && censusEnv != "" && censusEnv != "false") {
            var plan = new NuePlan();
            plan.architecture = meta.architecture;
            plan.layers = meta.numLayers;
            plan.heads = meta.numHeads;
            plan.kvHeads = meta.numKvHeads;
            plan.headDim = meta.headDim;
            plan.ropeNeox = rope.neox;
            plan.haxeMatmul = haxeMatmul;
            plan.kvQ8 = useKvQ8;
            plan.haxeFlash = useHaxeQ8;
            plan.requantLmHead = Sys.getEnvOr("NUE_REQUANT_LM_HEAD", "RAYZOR_REQUANT_LM_HEAD") != "0";
            plan.requantQ6K = Sys.getEnvOr("NUE_REQUANT_Q6K", "RAYZOR_REQUANT_Q6K") == "1";
            plan.poolWorkers = Q4Matmul.workerCount();
            plan.poolSpins = Q4Matmul.poolSpins();
            plan.poolRelax = Q4Matmul.poolRelax();
            plan.poolProfiling = Q4Matmul.poolProfiling();
            plan.poolAdaptive = Q4Matmul.poolAdaptive();
            plan.lmHeadScheme = schemeOrdinal(lmHead);
            plan.addEmbedding(schemeOrdinal(null));
            for (i in 0...blocks.length) {
                var attn = blocks[i].attn;
                plan.addLayer(i,
                    schemeOrdinal(attn.qProj), schemeOrdinal(attn.kProj), schemeOrdinal(attn.vProj),
                    cacheKind(attn.cache));
            }
            plan.addHead(plan.lmHeadScheme);
            plan.dump();
        }
        return model;
    }

    /** The scheme a projection's weights were built from, as the ordinal
        `NuePlan` records. A dense weight has no scheme. */
    static function schemeOrdinal(lin:Null<Linear>):Int {
        if (lin == null) return NuePlan.SCHEME_NONE;
        var qw = lin.qweight;
        if (qw == null) return NuePlan.SCHEME_NONE;
        return switch (qw.scheme()) {
            case INT8: NuePlan.SCHEME_INT8;
            case Q4_K_M: NuePlan.SCHEME_Q4_K_M;
            case Q6_K: NuePlan.SCHEME_Q6_K;
            case Q8_0: NuePlan.SCHEME_Q8_0;
        };
    }

    /** The cache a layer actually got, which is not always the one requested. */
    static function cacheKind(cache:KVCache):Int {
        if (cache.useQ8H) return NuePlan.CACHE_Q8_HAXE;
        if (cache.useQ8) return NuePlan.CACHE_Q8;
        return NuePlan.CACHE_F32;
    }

    static function buildBlock(
        meta:ModelMetadata, layerIndex:Int, dtype:DType,
        rope:RoPE, weights:NamedTensorMap, useKvQ8:Bool, useHaxeQ8:Bool,
        ?sp:rayzor.concurrent.SpinPool
    ):LlamaBlock {
        var prefix = "blk." + layerIndex + ".";
        var attnNorm = new RMSNorm(
            takeWeight(weights, prefix + "attn_norm.weight"),
            meta.normEps, "weight"
        );

        var cache = new KVCache(meta.maxSeqLen, meta.numKvHeads, meta.headDim, dtype,
            useKvQ8, useHaxeQ8, meta.numHeads);
        if (useKvQ8 && layerIndex == 0) {
            if (cache.useQ8H) {
                Sys.println("[kv-cache] Q8_0 mode enabled (guest cache + Haxe flash decode)");
            } else if (cache.useQ8) {
                Sys.println("[kv-cache] Q8_0 mode enabled");
            } else {
                Sys.println("[kv-cache] Q8_0 requested but head_dim=" + meta.headDim
                    + " is not a multiple of 32 — falling back to F32");
            }
        }
        // Attention-projection bias: absent in Llama/Mistral, present on
        // q/k/v (not o_proj) in Qwen2. Load conditionally so one builder
        // covers both families — GGUF stores the bias F32 even when the
        // weight is quantised, which `buildLinear` already handles.
        var qProj = buildLinear(weights, prefix + "attn_q.weight", biasIfPresent(weights, prefix + "attn_q"), sp);
        var kProj = buildLinear(weights, prefix + "attn_k.weight", biasIfPresent(weights, prefix + "attn_k"), sp);
        var vProj = buildLinear(weights, prefix + "attn_v.weight", biasIfPresent(weights, prefix + "attn_v"), sp);
        var oProj = buildLinear(weights, prefix + "attn_output.weight", biasIfPresent(weights, prefix + "attn_output"), sp);
        var attn = new GQAttention(
            qProj, kProj, vProj, oProj, rope, cache,
            meta.numHeads, meta.numKvHeads, meta.headDim
        );

        var ffnNorm = new RMSNorm(
            takeWeight(weights, prefix + "ffn_norm.weight"),
            meta.normEps, "weight"
        );

        var ffn = new SwiGLU(
            buildLinear(weights, prefix + "ffn_gate.weight", null, sp),
            buildLinear(weights, prefix + "ffn_up.weight", null, sp),
            buildLinear(weights, prefix + "ffn_down.weight", null, sp)
        );

        return new LlamaBlock(attnNorm, attn, ffnNorm, ffn, prefix);
    }

    /** Build a Linear from a weight name, picking the QTensor path when
     *  the loader registered a quantised entry. Bias is also F32 in GGUF
     *  even for Q4_K_M weights, so it's read via the F32 path. */
    static function buildLinear(weights:NamedTensorMap, weightName:String,
                                biasName:Null<String>,
                                ?sp:rayzor.concurrent.SpinPool):Linear {
        var b:Null<Tensor> = (biasName == null) ? null : weights.get(biasName);
        if (b != null && Sys.getEnvOr("NUE_DBG_BIAS", "") == "1") {
            Sys.println("[bias] " + biasName + " dtype=" + b.dtype() + " numel=" + b.numel()
                + " v=[" + b.getFlat(0) + ", " + b.getFlat(1) + ", " + b.getFlat(2) + "]");
        }
        if (weights.isQuantised(weightName)) {
            var qw = weights.getQuant(weightName);
            if (qw == null) {
                throw "nue.arch.Llama: missing quant weight '" + weightName + "'";
            }
            // Opt-in: re-quantise Q6_K matmul weights down to Q4_K_M so they
            // take the fast Q4 SDOT path instead of the slower Q6_K kernel.
            // Q4_K_M GGUFs promote accuracy-sensitive weights (ffn_down,
            // attn_v) to Q6_K; on the wasmtime host those go through the
            // SCALAR Q6_K dot (with a per-block dequant-scratch memset), and
            // in the browser/in-guest path through the heavier Q6_K SIMD
            // kernel — both far slower than Q4 SDOT. `requantQ6KToQ4KM()`
            // no-ops (returns null) on already-Q4 weights, so this hits only
            // the Q6_K ones. Trades a little quantisation accuracy on those
            // weights for speed — gated OFF by default; set RAYZOR_REQUANT_Q6K=1.
            if (Sys.getEnvOr("NUE_REQUANT_Q6K", "RAYZOR_REQUANT_Q6K") == "1") {
                var rq = qw.requantQ6KToQ4KM();
                if (rq != null) {
                    Sys.println("[requant] " + weightName + " Q6_K → Q4_K_M");
                    qw = rq;
                }
            }
            var lq = Linear.fromQuant(qw, b, "weight");
            lq.pool = sp;
            return lq;
        }
        var lf = new Linear(takeWeight(weights, weightName), b, "weight");
        lf.pool = sp;
        return lf;
    }

    /** `<base>.bias` when the GGUF carries it, else null. Llama-family
     *  checkpoints have no attention bias; Qwen2 biases q/k/v (not o_proj),
     *  so a single conditional covers both. */
    static function biasIfPresent(weights:NamedTensorMap, base:String):Null<String> {
        var bn = base + ".bias";
        return weights.exists(bn) ? bn : null;
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
