package nue.loader;

import sys.io.File;
import nue.Module;
import nue.arch.ArchRegistry;
import nue.model.ModelLoader;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import nue.loader.GGUFReader.MetaValue;
import nue.loader.GGUFTokenizer;
import nue.tokenizer.Tokenizer;
import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
import rayzor.ds.DType;

/**
 * GGUF (`*.gguf`) loader. Maps the on-disk format to the generic
 * `ModelMetadata` + tensor dictionary that `nue.arch.ArchRegistry`
 * consumes, so the same loader works across Llama / Mistral /
 * Qwen / Phi as long as a matching `ArchBuilder` is registered.
 *
 * GGUF naming is the canonical scheme everywhere in nue:
 * `token_embd.weight`, `blk.{L}.attn_q.weight`, etc. Other formats
 * (ONNX, safetensors) normalize to it in their loaders so the arch
 * builders don't need format awareness.
 *
 * **Status — supported tensor dtypes (v1):**
 *   - F32     (GGML 0)  — decoded inline via `Tensor.fromArray`.
 *   - F16     (GGML 1)  — widened to F32 via `Tensor.fromBytesF16`.
 *   - Q8_0    (GGML 8)  — dequantised to F32 via `Tensor.fromBytesQ8_0`.
 *   - Q4_K_M  (GGML 14) — loaded as `QTensor.fromBytesQ4KM`, then
 *                         dequantised to F32 via `qt.dequant()` so it
 *                         drops into the existing `Tensor`-only
 *                         `NamedTensorMap` / `Linear` plumbing. The
 *                         compressed path (keeping weights as QTensor
 *                         and routing through `QTensor.matmulF32`) is
 *                         Phase 4b — requires `Linear` to accept a
 *                         polymorphic weight.
 *
 * The metadata + tensor-info parsing is fully functional for every
 * dtype today.
 */
class GGUFLoader implements ModelLoader {
    public function new() {}

    public function readMetadata(path:String):ModelMetadata {
        var bytes = File.getBytes(path);
        var reader = new GGUFReader(bytes);
        return metadataFromReader(reader);
    }

    public function readNamedTensors(path:String):NamedTensorMap {
        var bytes = File.getBytes(path);
        var reader = new GGUFReader(bytes);
        return tensorsFromReader(reader);
    }

    public var registry:ArchRegistry;

    public function load(path:String):Module {
        var bytes = File.getBytes(path);
        var reader = new GGUFReader(bytes);
        var meta = metadataFromReader(reader);
        var weights = tensorsFromReader(reader);
        var reg = (registry != null) ? registry : ArchRegistry.withDefaults();
        return reg.build(meta, weights);
    }

    /**
     * Pull the embedded vocab + merges out of the GGUF metadata and
     * build a `BPETokenizer`. Most GGUFs in circulation (Llama 1/2/3,
     * Mistral, Qwen, …) carry the tokenizer inline so a single
     * `.gguf` file is enough to encode prompts and decode generations.
     */
    public function tokenizer(path:String):Tokenizer {
        var bytes = File.getBytes(path);
        var reader = new GGUFReader(bytes);
        return GGUFTokenizer.build(reader);
    }

    /**
     * Open a GGUF once and return the model + tokenizer together.
     * Cheaper than calling `load(path)` + `tokenizer(path)` separately
     * (which would parse the header twice).
     */
    public function loadWithTokenizer(path:String):LoadedModel {
        trace("[lwt] 1");
        var bytes = File.getBytes(path);
        trace("[lwt] 2");
        var reader = new GGUFReader(bytes);
        trace("[lwt] 3");
        var meta = metadataFromReader(reader);
        trace("[lwt] 4");
        var weights = tensorsFromReader(reader);
        trace("[lwt] 5");
        var reg = (registry != null) ? registry : ArchRegistry.withDefaults();
        trace("[lwt] 6");
        var model = reg.build(meta, weights);
        trace("[lwt] 7");
        var tok = GGUFTokenizer.build(reader);
        trace("[lwt] 8");
        return { model: model, tokenizer: tok, metadata: meta };
    }

    /**
     * Translate the GGUF reader's raw KV map into the generic
     * `ModelMetadata` schema. Field names follow the GGUF spec —
     * `llama.attention.head_count`, `llama.embedding_length`, etc.
     * Each architecture family uses the same prefix as its name in
     * `general.architecture`, so we read with that dynamic prefix.
     */
    private static function metadataFromReader(reader:GGUFReader):ModelMetadata {
        var arch = reader.metaString("general.architecture");
        var p = arch + ".";
        var hiddenSize = reader.metaInt(p + "embedding_length");
        var numLayers = reader.metaInt(p + "block_count");
        var numHeads = reader.metaInt(p + "attention.head_count");
        // num_kv_heads is optional — falls back to numHeads for MHA.
        var numKvHeads = readIntOr(reader, p + "attention.head_count_kv", numHeads);
        var headDim = Std.int(hiddenSize / numHeads);
        var maxSeqLen = readIntOr(reader, p + "context_length", 2048);
        var intermediateSize = readIntOr(reader, p + "feed_forward_length", hiddenSize * 4);
        var vocabSize = readIntOr(reader, p + "vocab_size", 0);
        if (vocabSize == 0) {
            vocabSize = readIntOr(reader, "tokenizer.ggml.tokens.count", 32000);
        }
        var normEps = readFloatOr(reader, p + "attention.layer_norm_rms_epsilon", 1e-5);
        var ropeBase = readFloatOr(reader, p + "rope.freq_base", 10000.0);
        var tieEmbed = readBoolOr(reader, p + "tie_word_embeddings", false);

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

    private static function tensorsFromReader(reader:GGUFReader):NamedTensorMap {
        var result = new NamedTensorMap();
        var infos:Array<GGUFReader.TensorInfo> = reader.tensorInfos;
        for (info in infos) {
            var raw = reader.tensorBytes(info);
            populateTensor(result, raw, info);
        }
        return result;
    }

    /** Lookup wrapper used by metadata helpers. */
    private static inline function getMeta(reader:GGUFReader, key:String):GGUFReader.MetaValue {
        return reader.findMeta(key);
    }

    /**
     * Decode a single tensor's raw bytes and register it in the result
     * map. Q4_K_M weights go in as QTensors (no dequant, no F32 copy);
     * other dtypes go in as F32/F16/Q8_0 `Tensor` values.
     */
    private static function populateTensor(
        result:NamedTensorMap, raw:haxe.io.Bytes, info:GGUFReader.TensorInfo
    ):Void {
        switch (info.dtype) {
            case 0: // F32
                var nElem = 1;
                for (d in info.dims) nElem *= d;
                var floats:Array<Float> = [];
                var pos = 0;
                for (_ in 0...nElem) {
                    floats.push(raw.getFloat(pos));
                    pos += 4;
                }
                var t = Tensor.fromArray(floats, F32).reshape(info.dims);
                result.set(info.name, t);
            case 1: // F16
                result.set(info.name, Tensor.fromBytesF16(raw, info.dims));
            case 8: // Q8_0
                result.set(info.name, Tensor.fromBytesQ8_0(raw, info.dims));
            case 12, 14: // Q4_K / Q4_K_M — Phase 4b: keep compressed.
                result.setQuant(info.name, decodeQ4KMRaw(raw, info));
            case _:
                throw "GGUFLoader: GGML dtype " + info.dtype + " not implemented (tensor '" + info.name + "').";
        }
    }

    /** Build a Q4_K_M `QTensor` view over the file's raw bytes without
     *  dequantising. The QTensor has `rows = out = dims[last]` and
     *  `cols = in = product of earlier dims` — the PyTorch `[out, in]`
     *  orientation that `Linear.fromQuant` expects. */
    private static function decodeQ4KMRaw(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):QTensor {
        var rows = info.dims[info.dims.length - 1];
        var cols = 1;
        for (i in 0...info.dims.length - 1) cols *= info.dims[i];
        var qt = QTensor.fromBytesQ4KM(raw, rows, cols);
        if (qt == null) {
            throw "GGUFLoader: QTensor.fromBytesQ4KM returned null for '"
                + info.name + "' (rows=" + rows + ", cols=" + cols + ").";
        }
        return qt;
    }

    /** Legacy F32-dequant path, retained for diagnostic / regression use.
     *  Not on the hot inference path — `populateTensor` ships QTensors
     *  through directly. */
    private static function decodeQ4KM(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):Tensor {
        var qt = decodeQ4KMRaw(raw, info);
        var dq = qt.dequant();
        qt.free();
        return (info.dims.length == 2) ? dq : dq.reshape(info.dims);
    }

    private static function readIntOr(reader:GGUFReader, key:String, fallback:Int):Int {
        var v = reader.findMeta(key);
        if (v == null) return fallback;
        return switch (v) {
            case U8(x) | I8(x) | U16(x) | I16(x) | U32(x) | I32(x) | U64(x) | I64(x): x;
            case _: fallback;
        };
    }

    private static function readFloatOr(reader:GGUFReader, key:String, fallback:Float):Float {
        var v = reader.findMeta(key);
        if (v == null) return fallback;
        return switch (v) {
            case F32(x) | F64(x): x;
            case U32(x) | I32(x): x;
            case _: fallback;
        };
    }

    private static function readBoolOr(reader:GGUFReader, key:String, fallback:Bool):Bool {
        var v = reader.findMeta(key);
        if (v == null) return fallback;
        return switch (v) {
            case Bool(b): b;
            case _: fallback;
        };
    }
}

typedef LoadedModel = {
    var model:Module;
    var tokenizer:Tokenizer;
    var metadata:ModelMetadata;
};
