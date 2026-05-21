package nue.loader;

import haxe.Json;
import haxe.io.Bytes;
import sys.io.File;
import sys.FileSystem;
import nue.Module;
import nue.arch.ArchRegistry;
import nue.model.ModelLoader;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import rayzor.ds.Tensor;

/**
 * Safetensors (`*.safetensors`) loader — Hugging Face's preferred
 * format for sharing trained weights. Layout:
 *
 * ```
 *   u64 header_len             (little-endian)
 *   header_len bytes of JSON   (per-tensor names, shapes, dtypes,
 *                               byte ranges into the data section)
 *   tensor data                (raw bytes, contiguous, aligned)
 * ```
 *
 * Compared to GGUF: no quantisation, no embedded tokenizer, no
 * architecture hint. Quantised weights are out of scope (the format
 * doesn't carry block scales). Architecture metadata lives in a
 * companion `config.json` (HF convention) that this loader reads
 * when present.
 *
 * **Supported tensor dtypes (v1):**
 *   - `F32` — copied straight through.
 *   - `F16` — widened to F32 via `Tensor.fromBytesF16`.
 *
 * BF16, integer, and bool tensors throw — they're rare in LLM
 * checkpoints (BF16 is the main gap; add via `fromBytesBF16` when
 * needed). The error message names the dtype + tensor.
 *
 * **HF → GGUF name normalisation.** HF names (`model.embed_tokens.weight`,
 * `model.layers.{L}.self_attn.q_proj.weight`, ...) are translated to
 * the GGUF canonical scheme that all `nue.arch.*` builders read
 * against (`token_embd.weight`, `blk.{L}.attn_q.weight`, ...). The
 * translation table is best-effort — names that don't match any
 * pattern pass through unchanged so user code can extend coverage
 * without forking the loader.
 */
class SafetensorsLoader implements ModelLoader {
    public var registry:ArchRegistry;

    public function new() {}

    public function readMetadata(path:String):ModelMetadata {
        var configPath = companionConfigPath(path);
        if (!FileSystem.exists(configPath)) {
            throw "SafetensorsLoader: no companion config.json found at '" + configPath
                + "'. HuggingFace safetensors checkpoints need the config.json sibling for "
                + "architecture metadata. Either drop the config in place or call "
                + "readNamedTensors(path) and build the ModelMetadata yourself.";
        }
        var raw = File.getContent(configPath);
        var cfg:Dynamic = Json.parse(raw);
        return metadataFromConfig(cfg);
    }

    public function readNamedTensors(path:String):NamedTensorMap {
        var bytes = File.getBytes(path);
        return parseTensors(bytes);
    }

    public function load(path:String):Module {
        var meta = readMetadata(path);
        var weights = readNamedTensors(path);
        var reg = (registry != null) ? registry : ArchRegistry.withDefaults();
        return reg.build(meta, weights);
    }

    // ------------------------------------------------------------------
    // Header parsing
    // ------------------------------------------------------------------

    static function parseTensors(bytes:Bytes):NamedTensorMap {
        // Header length is a little-endian u64 at offset 0. Safetensors
        // headers are tiny (KBs) so the top 32 bits are 0; we read the
        // low int32.
        var headerLen = bytes.getInt32(0);
        var headerJson = bytes.getString(8, headerLen);
        var header:Dynamic = Json.parse(headerJson);
        var dataStart = 8 + headerLen;

        var result = new NamedTensorMap();
        var fields = Reflect.fields(header);
        for (key in fields) {
            if (key == "__metadata__") continue;
            var entry:Dynamic = Reflect.field(header, key);
            var dtype:String = entry.dtype;
            var shape:Array<Int> = entry.shape;
            var offsets:Array<Int> = entry.data_offsets;
            var byteStart = dataStart + offsets[0];
            var byteEnd = dataStart + offsets[1];
            var slice = bytes.sub(byteStart, byteEnd - byteStart);
            var tensor = decodeTensor(dtype, slice, shape, key);
            var canonical = normaliseName(key);
            result.set(canonical, tensor);
        }
        return result;
    }

    static function decodeTensor(dtype:String, raw:Bytes, shape:Array<Int>, name:String):Tensor {
        return switch (dtype) {
            case "F32":
                var nElem = 1;
                for (d in shape) nElem *= d;
                var floats:Array<Float> = [];
                var pos = 0;
                for (_ in 0...nElem) {
                    floats.push(raw.getFloat(pos));
                    pos += 4;
                }
                var t = Tensor.fromArray(floats, F32);
                t.reshape(shape);
            case "F16":
                Tensor.fromBytesF16(raw, shape);
            case "BF16":
                throw "SafetensorsLoader: BF16 not yet supported (tensor '" + name
                    + "'). Add a `Tensor.fromBytesBF16` extern when this dtype "
                    + "appears in a checkpoint we need to load.";
            case _:
                throw "SafetensorsLoader: dtype " + dtype + " not implemented "
                    + "(tensor '" + name + "').";
        }
    }

    // ------------------------------------------------------------------
    // Name normalisation: HuggingFace → GGUF
    // ------------------------------------------------------------------

    static function normaliseName(hfName:String):String {
        // Top-level singletons.
        if (hfName == "model.embed_tokens.weight") return "token_embd.weight";
        if (hfName == "model.norm.weight") return "output_norm.weight";
        if (hfName == "lm_head.weight") return "output.weight";

        // BERT-style top-level.
        if (hfName == "embeddings.word_embeddings.weight") return "token_embd.weight";
        if (hfName == "embeddings.position_embeddings.weight") return "position_embd.weight";
        if (hfName == "embeddings.token_type_embeddings.weight") return "token_types.weight";
        if (hfName == "embeddings.LayerNorm.weight") return "token_embd_norm.weight";
        if (hfName == "embeddings.LayerNorm.bias") return "token_embd_norm.bias";

        // Per-layer Llama / Mistral / Qwen pattern:
        //   model.layers.{L}.input_layernorm.weight     → blk.{L}.attn_norm.weight
        //   model.layers.{L}.self_attn.q_proj.weight    → blk.{L}.attn_q.weight
        //   model.layers.{L}.self_attn.k_proj.weight    → blk.{L}.attn_k.weight
        //   model.layers.{L}.self_attn.v_proj.weight    → blk.{L}.attn_v.weight
        //   model.layers.{L}.self_attn.o_proj.weight    → blk.{L}.attn_output.weight
        //   model.layers.{L}.post_attention_layernorm.weight → blk.{L}.ffn_norm.weight
        //   model.layers.{L}.mlp.gate_proj.weight       → blk.{L}.ffn_gate.weight
        //   model.layers.{L}.mlp.up_proj.weight         → blk.{L}.ffn_up.weight
        //   model.layers.{L}.mlp.down_proj.weight       → blk.{L}.ffn_down.weight
        if (StringTools.startsWith(hfName, "model.layers.")) {
            var rest = hfName.substr("model.layers.".length);
            var dot = rest.indexOf(".");
            if (dot < 0) return hfName;
            var layerIdx = rest.substr(0, dot);
            var suffix = rest.substr(dot + 1);
            var translated = translateLayerSuffix(suffix);
            if (translated == null) return hfName;
            return "blk." + layerIdx + "." + translated;
        }

        // BERT per-layer: encoder.layer.{L}.{...} → blk.{L}.{...}
        if (StringTools.startsWith(hfName, "encoder.layer.")) {
            var rest = hfName.substr("encoder.layer.".length);
            var dot = rest.indexOf(".");
            if (dot < 0) return hfName;
            var layerIdx = rest.substr(0, dot);
            var suffix = rest.substr(dot + 1);
            var translated = translateBertLayerSuffix(suffix);
            if (translated == null) return hfName;
            return "blk." + layerIdx + "." + translated;
        }

        return hfName;
    }

    static function translateLayerSuffix(suffix:String):String {
        // Llama-family.
        return switch (suffix) {
            case "input_layernorm.weight": "attn_norm.weight";
            case "post_attention_layernorm.weight": "ffn_norm.weight";
            case "self_attn.q_proj.weight": "attn_q.weight";
            case "self_attn.k_proj.weight": "attn_k.weight";
            case "self_attn.v_proj.weight": "attn_v.weight";
            case "self_attn.o_proj.weight": "attn_output.weight";
            case "mlp.gate_proj.weight": "ffn_gate.weight";
            case "mlp.up_proj.weight": "ffn_up.weight";
            case "mlp.down_proj.weight": "ffn_down.weight";
            case _: null;
        };
    }

    static function translateBertLayerSuffix(suffix:String):String {
        return switch (suffix) {
            case "attention.self.query.weight": "attn_q.weight";
            case "attention.self.query.bias": "attn_q.bias";
            case "attention.self.key.weight": "attn_k.weight";
            case "attention.self.key.bias": "attn_k.bias";
            case "attention.self.value.weight": "attn_v.weight";
            case "attention.self.value.bias": "attn_v.bias";
            case "attention.output.dense.weight": "attn_output.weight";
            case "attention.output.dense.bias": "attn_output.bias";
            case "attention.output.LayerNorm.weight": "attn_norm.weight";
            case "attention.output.LayerNorm.bias": "attn_norm.bias";
            case "intermediate.dense.weight": "ffn_up.weight";
            case "intermediate.dense.bias": "ffn_up.bias";
            case "output.dense.weight": "ffn_down.weight";
            case "output.dense.bias": "ffn_down.bias";
            case "output.LayerNorm.weight": "ffn_norm.weight";
            case "output.LayerNorm.bias": "ffn_norm.bias";
            case _: null;
        };
    }

    // ------------------------------------------------------------------
    // Companion config.json
    // ------------------------------------------------------------------

    static function companionConfigPath(safetensorsPath:String):String {
        // Same directory, file named `config.json` — the HF convention.
        var lastSlash = safetensorsPath.lastIndexOf("/");
        var dir = (lastSlash >= 0) ? safetensorsPath.substr(0, lastSlash + 1) : "";
        return dir + "config.json";
    }

    static function metadataFromConfig(cfg:Dynamic):ModelMetadata {
        var arch = inferArch(cfg);
        var hiddenSize:Int = fieldInt(cfg, "hidden_size", 0);
        var numLayers:Int = fieldInt(cfg, "num_hidden_layers", fieldInt(cfg, "n_layer", 0));
        var numHeads:Int = fieldInt(cfg, "num_attention_heads", fieldInt(cfg, "n_head", 0));
        var numKvHeads:Int = fieldInt(cfg, "num_key_value_heads", numHeads);
        var headDim:Int = (numHeads > 0) ? Std.int(hiddenSize / numHeads) : 0;
        var maxSeqLen:Int = fieldInt(cfg, "max_position_embeddings", 2048);
        var intermediateSize:Int = fieldInt(cfg, "intermediate_size", hiddenSize * 4);
        var vocabSize:Int = fieldInt(cfg, "vocab_size", 32000);
        var normEps:Float = fieldFloat(cfg, "rms_norm_eps",
            fieldFloat(cfg, "layer_norm_eps", 1e-5));
        var ropeBase:Float = fieldFloat(cfg, "rope_theta", 10000.0);
        var tieEmbed:Bool = fieldBool(cfg, "tie_word_embeddings", false);

        return {
            architecture: arch,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numLayers: numLayers,
            numHeads: numHeads,
            numKvHeads: numKvHeads,
            headDim: headDim,
            vocabSize: vocabSize,
            maxSeqLen: maxSeqLen,
            normEps: normEps,
            ropeBase: ropeBase,
            tieWordEmbeddings: tieEmbed
        };
    }

    static function inferArch(cfg:Dynamic):String {
        // HF `model_type` is the canonical hint. Map to nue arch names.
        var mt:Dynamic = Reflect.field(cfg, "model_type");
        if (mt == null) return "llama"; // sensible default for safetensors
        var s:String = mt;
        return switch (s) {
            case "llama" | "mistral" | "qwen2": "llama";
            case "bert" | "roberta" | "deberta-v2": "bert";
            case _: s;
        };
    }

    static function fieldInt(o:Dynamic, key:String, fallback:Int):Int {
        var v:Dynamic = Reflect.field(o, key);
        if (v == null) return fallback;
        return Std.int(v);
    }

    static function fieldFloat(o:Dynamic, key:String, fallback:Float):Float {
        var v:Dynamic = Reflect.field(o, key);
        if (v == null) return fallback;
        return v;
    }

    static function fieldBool(o:Dynamic, key:String, fallback:Bool):Bool {
        var v:Dynamic = Reflect.field(o, key);
        if (v == null) return fallback;
        return v;
    }
}
