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
import nue.tokenizer.BPETokenizer;
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
    public function tokenizer(path:String):BPETokenizer {
        var bytes = File.getBytes(path);
        var reader = new GGUFReader(bytes);
        return GGUFTokenizer.build(reader);
    }

    /** Truthy-env check: ON iff the var is set to a value that isn't
     *  `0`/`false`/empty (so `=0` disables, matching the rest of the CLI). */
    static function _profEnvOn(name:String):Bool {
        var v = Sys.getEnv(name);
        if (v == null) return false;
        var lv = v.toLowerCase();
        return lv != "0" && lv != "" && lv != "false" && lv != "no";
    }

    /**
     * Open a GGUF once and return the model + tokenizer together.
     * Cheaper than calling `load(path)` + `tokenizer(path)` separately
     * (which would parse the header twice).
     */
    public function loadWithTokenizer(path:String, maxCtx:Int = 0):LoadedModel {
        // VALUE-gated: `=0`/`=false`/empty/unset mean OFF (not `!= null`).
        var profileOn = _profEnvOn("RAYZOR_PROFILE_LOAD")
            || _profEnvOn("RAYZOR_PROFILE_DECODE");
        var totalStart = profileOn ? Sys.time() : 0.0;
        var phaseStart = totalStart;
        var fileS = 0.0;
        var readerS = 0.0;
        var metaS = 0.0;
        var tensorsS = 0.0;
        var registryS = 0.0;
        var buildS = 0.0;
        var tokenizerS = 0.0;
        var bytes = File.getBytes(path);
        if (profileOn) {
            fileS = Sys.time() - phaseStart;
            phaseStart = Sys.time();
        }
        var reader = new GGUFReader(bytes);
        if (profileOn) {
            readerS = Sys.time() - phaseStart;
            phaseStart = Sys.time();
        }
        var meta = metadataFromReader(reader);
        if (maxCtx > 0 && maxCtx < meta.maxSeqLen) {
            meta.maxSeqLen = maxCtx;
        }
        if (profileOn) {
            metaS = Sys.time() - phaseStart;
            phaseStart = Sys.time();
        }
        var weights = tensorsFromReader(reader);
        if (profileOn) {
            tensorsS = Sys.time() - phaseStart;
            phaseStart = Sys.time();
        }
        var reg = (registry != null) ? registry : ArchRegistry.withDefaults();
        if (profileOn) {
            registryS = Sys.time() - phaseStart;
            phaseStart = Sys.time();
        }
        var model = reg.build(meta, weights);
        if (profileOn) {
            buildS = Sys.time() - phaseStart;
            phaseStart = Sys.time();
        }
        var tok = GGUFTokenizer.build(reader);
        // The top-level file handle is dead now: tensors copied their slices,
        // and QTensors wrap mmap-backed tensor views zero-copy. Releasing this
        // handle drops one owner reference; the mmap itself stays alive through
        // those QTensor views for native inference. On wasm this still matters:
        // owned linear-memory file buffers are reclaimed instead of sitting as
        // ~0.77 GB of dead heap. Default-on; set RAYZOR_FREE_GGUF_BYTES=0 to opt out.
        if (Sys.getEnv("RAYZOR_FREE_GGUF_BYTES") != "0") {
            bytes.free();
        }
        if (profileOn) {
            tokenizerS = Sys.time() - phaseStart;
            trace("[profile-load] file_s=" + fileS
                + " reader_s=" + readerS
                + " metadata_s=" + metaS
                + " tensors_s=" + tensorsS
                + " registry_s=" + registryS
                + " build_s=" + buildS
                + " tokenizer_s=" + tokenizerS
                + " total_s=" + (Sys.time() - totalStart));
        }
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
            case 0: // F32 — memcpy bytes directly into a fresh F32 tensor.
                    // Avoids the `Array<Float>.push` precision-loss bug
                    // (boxes via i64) when crossing the runtime boundary.
                result.set(info.name, Tensor.fromBytesF32(raw, info.dims));
            case 1: // F16
                result.set(info.name, Tensor.fromBytesF16(raw, info.dims));
            case 8: // Q8_0
                result.set(info.name, Tensor.fromBytesQ8_0(raw, info.dims));
            case 12: // Q4_K — 144-byte super-blocks; the dominant Q4_K_M weight scheme.
                result.setQuant(info.name, decodeQ4KMRaw(raw, info));
            case 14: // Q6_K — 210-byte super-blocks. Used in Q4_K_M variants for
                     // token_embd, attn_v, and ffn_down (where K_M means "mixed":
                     // some weights get Q6_K to preserve accuracy).
                result.setQuant(info.name, decodeQ6KRaw(raw, info));
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

    /** Same as `decodeQ4KMRaw` but for Q6_K (GGUF dtype 14) — 210-byte
     *  super-blocks of 256 weights. Used for token_embd, attn_v, and
     *  ffn_down in `Q4_K_M` variant GGUFs (the "M" / "mixed" suffix
     *  promotes accuracy-sensitive weights from Q4_K up to Q6_K). */
    private static function decodeQ6KRaw(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):QTensor {
        var rows = info.dims[info.dims.length - 1];
        var cols = 1;
        for (i in 0...info.dims.length - 1) cols *= info.dims[i];
        var qt = QTensor.fromBytesQ6K(raw, rows, cols);
        if (qt == null) {
            throw "GGUFLoader: QTensor.fromBytesQ6K returned null for '"
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
    var tokenizer:BPETokenizer;
    var metadata:ModelMetadata;
};
