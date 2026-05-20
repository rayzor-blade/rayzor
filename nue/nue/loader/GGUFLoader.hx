package nue.loader;

import sys.io.File;
import nue.Module;
import nue.arch.ArchRegistry;
import nue.model.ModelLoader;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import nue.loader.GGUFReader.MetaValue;
import rayzor.ds.Tensor;
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
 *   - F32 (GGML type 0)  — decoded inline via `Tensor.fromArray`.
 *   - F16 (GGML type 1)  — TODO (need `Tensor.fromBytesF16` extern).
 *   - Q4_K_M (type 14)   — TODO (wire to `QTensor.wrapQ4KM`; the
 *                          runtime kernels are in place, the
 *                          loader-side wrapper isn't yet).
 *   - Q8_0 (type 8)      — TODO (similar to Q4_K_M).
 *
 * The metadata + tensor-info parsing is fully functional for every
 * dtype today — only the actual tensor-data materialisation paths
 * are dtype-gated. A model with mostly F32 weights will load cleanly;
 * the Llama 3.2 1B Q4_K_M demo is blocked on the QTensor wiring.
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
        for (info in reader.tensorInfos) {
            var raw = reader.tensorBytes(info);
            var tensor = decodeTensor(raw, info);
            result.set(info.name, tensor);
        }
        return result;
    }

    /** Lookup wrapper used by metadata helpers — adapts the reader's
        parallel-array `findMeta` API to the previous Map-style call. */
    private static inline function getMeta(reader:GGUFReader, key:String):GGUFReader.MetaValue {
        return reader.findMeta(key);
    }

    /**
     * Decode a single tensor's raw bytes into a `Tensor`. F32 is the
     * only dtype materialised inline today. Other dtypes throw a
     * descriptive error pointing at the missing extern — keeps the
     * "unsupported" failure mode loud rather than silent.
     */
    private static function decodeTensor(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):Tensor {
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
                var t = Tensor.fromArray(floats, F32);
                return t.reshape(info.dims);
            case 1:
                throw "GGUFLoader: F16 tensors not yet supported — needs `Tensor.fromBytesF16` extern. Tensor '"
                    + info.name + "' has dtype F16.";
            case 8:
                throw "GGUFLoader: Q8_0 tensors not yet supported — wire to `QTensor.wrapQ8` runtime entry. Tensor '"
                    + info.name + "'.";
            case 12, 14:
                throw "GGUFLoader: Q4_K / Q4_K_M tensors not yet wired in the loader — wire to `QTensor.wrapQ4KM`. Tensor '"
                    + info.name + "' has dtype Q4_K.";
            case _:
                throw "GGUFLoader: GGML dtype " + info.dtype + " not implemented (tensor '" + info.name + "').";
        }
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
