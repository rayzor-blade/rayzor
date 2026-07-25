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
import nue.engine.PrefillGraph;

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
 * **Status — supported tensor dtypes:**
 *   - F32     (GGML 0)  — memcpy'd via `Tensor.fromBytesF32`.
 *   - F16     (GGML 1)  — widened to F32 via `Tensor.fromBytesF16`.
 *   - Q5_0    (GGML 6)  — converted at load to an INT8 per-row `QTensor`
 *                         (`QTensor.fromBytesQ5_0Int8`); covers models
 *                         whose inner dims defeat the 256-wide k-quants.
 *   - Q5_1    (GGML 7)  — as Q5_0, plus an explicit per-block min.
 *   - Q5_K    (GGML 13) — re-encoded to Q4_K_M at load (shared super-block
 *                         geometry; smaller than the source and on the
 *                         fastest kernel).
 *   - Q8_0    (GGML 8)  — dequantised to F32 via `Tensor.fromBytesQ8_0`.
 *   - Q4_K    (GGML 12) — zero-copy `QTensor.fromBytesQ4KM`.
 *   - Q6_K    (GGML 14) — zero-copy `QTensor.fromBytesQ6K`.
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
        var model = reg.build(meta, weights);
        applyEngines(model, meta, reader, path);
        return model;
    }

    /** Post-build engine wiring, shared by `load` and `loadWithTokenizer`:
     * bert pooling convention (`bert.pooling_type`: 1 mean, 2 CLS) and the
     * llama fused prefill graph (auto-detect `<stem>.prefill_s{S}.mlmodelc`
     * next to the gguf — capability IS the config; NUE_PREFILL=off kills).
     */
    function applyEngines(model:Module, meta:ModelMetadata, reader:GGUFReader, path:String):Void {
        if (meta.architecture == "bert") {
            var pooling = readIntOr(reader, "bert.pooling_type", 1);
            if (pooling == 2) cast(model, nue.arch.BertModel).clsPool = true;
        }
        // Llama-family (Llama/Mistral/Qwen2) all build a LlamaModel and share
        // the same prefill-graph + decode-warm wiring.
        if (meta.architecture == "llama" || meta.architecture == "qwen2") {
            var lm = cast(model, nue.arch.LlamaModel);
            // Fused CoreML prefill graph — auto-detected next to the gguf.
            // NUE_PREFILL=off drops it entirely (pure-CPU per-op / morselized
            // prefill). This ONLY affects prefill/ttft; decode is CPU either way.
            var kill = Sys.getEnv("NUE_PREFILL");
            if (kill != "off" && kill != "0") {
                var slash = path.lastIndexOf("/");
                var dir = slash >= 0 ? path.substr(0, slash) : ".";
                var file = slash >= 0 ? path.substr(slash + 1) : path;
                var dot = file.lastIndexOf(".gguf");
                var stem = dot > 0 ? file.substr(0, dot) : file;
                var db = rayzor.Bytes.ofString(dir);
                var sb = rayzor.Bytes.ofString(stem);
                var handle:Int = PrefillGraph.load(db.address(), db.length,
                    sb.address(), sb.length, meta.hiddenSize, meta.numLayers,
                    meta.numKvHeads, meta.headDim);
                db.free();
                sb.free();
                if (handle > 0) {
                    lm.prefillHandle = handle;
                    Sys.println("[prefill] engine=graph on");
                }
            }
            // Warm the forward + DECODE kernels at load, INDEPENDENT of the graph
            // so a pure-CPU server (NUE_PREFILL=off) also gets a warm decode from
            // request 1 (the bundle tier-promotes by call count — an unwarmed
            // first request promotes mid-generation, the low-floor requests).
            // Default ON; opt out with NUE_DECODE_WARM=0. With the graph on this
            // also pays its one-time E5RT specialization here, not on first prefill.
            var noWarm = Sys.getEnvOr("NUE_DECODE_WARM", "RAYZOR_DECODE_WARM") == "0";
            if (!noWarm) {
                var tW = Sys.time();
                lm.warm();
                Sys.println("[warm] model warmed in " + (Sys.time() - tW) + "s");
            }
        }
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

    /** Truthy-env check: ON iff the var is set to a value that isn't
     *  `0`/`false`/empty (so `=0` disables, matching the rest of the CLI). */
    static function _profEnvOn(name:String, alias:String):Bool {
        var v = Sys.getEnvOr(name, alias);
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
        var profileOn = _profEnvOn("NUE_PROFILE_LOAD", "RAYZOR_PROFILE_LOAD")
            || _profEnvOn("NUE_PROFILE_DECODE", "RAYZOR_PROFILE_DECODE");
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
        applyEngines(model, meta, reader, path);
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
        if (Sys.getEnvOr("NUE_FREE_GGUF_BYTES", "RAYZOR_FREE_GGUF_BYTES") != "0") {
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
        // Vocab: the `<arch>.vocab_size` key is often absent (Qwen2 omits it),
        // so derive from the `token_embd.weight` tensor shape — the embedding
        // table is [vocab, hidden], authoritative regardless of metadata keys.
        // Only fall back to the tokenizer count / a default if no embedding
        // tensor is present (e.g. an encoder head-only checkpoint).
        var vocabSize = readIntOr(reader, p + "vocab_size", 0);
        if (vocabSize == 0) vocabSize = vocabFromEmbd(reader, hiddenSize);
        if (vocabSize == 0) vocabSize = readIntOr(reader, "tokenizer.ggml.tokens.count", 32000);
        // LayerNorm epsilon: BERT/GPT-family store the non-RMS key
        // `attention.layer_norm_epsilon` (1e-12 for BERT); Llama-family
        // store `attention.layer_norm_rms_epsilon` (1e-5). Try the non-RMS
        // key first, then RMS, then an arch-appropriate default.
        var normEps = readFloatOr(reader, p + "attention.layer_norm_epsilon",
            readFloatOr(reader, p + "attention.layer_norm_rms_epsilon",
                (arch == "bert" ? 1e-12 : 1e-5)));
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

    /** Vocab size from the `token_embd.weight` shape — one dim is the hidden
     *  size, the other is the vocabulary. Authoritative when the metadata
     *  omits `<arch>.vocab_size` (Qwen2). Returns 0 if no embedding tensor. */
    private static function vocabFromEmbd(reader:GGUFReader, hiddenSize:Int):Int {
        for (info in reader.tensorInfos) {
            if (info.name == "token_embd.weight") {
                for (d in info.dims) {
                    if (d != hiddenSize) return d;
                }
            }
        }
        return 0;
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
        // Orient multi-dim tensors to [out, in] — the convention the quant
        // path and both weight consumers (Linear.matmulT, Embedding.gatherRows)
        // use. GGUF stores ggml dims [in, out] with bytes row-major over the
        // output dim, so reshaping to [out, in] needs no data move.
        var dims = orientDims(info.dims);
        switch (info.dtype) {
            case 0: // F32 — memcpy bytes directly into a fresh F32 tensor.
                    // Avoids the `Array<Float>.push` precision-loss bug
                    // (boxes via i64) when crossing the runtime boundary.
                result.set(info.name, Tensor.fromBytesF32(raw, dims));
            case 1: // F16
                result.set(info.name, Tensor.fromBytesF16(raw, dims));
            case 6: // Q5_0 — 22-byte blocks of 32. Carries the tensors the
                    // 256-wide k-quants can't express (inner dims not
                    // divisible by 256, e.g. Qwen2-0.5B hidden 896). Loaded
                    // as an INT8 per-row QTensor so the matmul stays on a
                    // quantised integer-dot path instead of an 8× F32 copy.
                result.setQuant(info.name, decodeQ5_0Int8(raw, info));
            case 7: // Q5_1 — 24-byte blocks of 32. Like Q5_0 but with an
                    // explicit min per block; same INT8 landing for the same
                    // reason (a 32-element block cannot ride the 256-wide
                    // k-quant path).
                result.setQuant(info.name, decodeQ5_1Int8(raw, info));
            case 8: // Q8_0
                result.set(info.name, Tensor.fromBytesQ8_0(raw, dims));
            case 12: // Q4_K — 144-byte super-blocks; the dominant Q4_K_M weight scheme.
                result.setQuant(info.name, decodeQ4KMRaw(raw, info));
            case 13: // Q5_K — 176-byte super-blocks. Re-encoded to Q4_K_M (same
                     // super-block geometry, fastest kernel, and 4.5 bits/weight
                     // vs Q5_K's 5.5 — so the weights SHRINK rather than grow).
                result.setQuant(info.name, decodeQ5KToQ4KM(raw, info));
            case 14: // Q6_K — 210-byte super-blocks. Used in Q4_K_M variants for
                     // token_embd, attn_v, and ffn_down (where K_M means "mixed":
                     // some weights get Q6_K to preserve accuracy).
                result.setQuant(info.name, decodeQ6KRaw(raw, info));
            case _:
                throw "GGUFLoader: GGML dtype " + info.dtype + " not implemented (tensor '" + info.name + "').";
        }
    }

    /** Orient raw GGUF dims to the `[out, in]` (PyTorch row-major)
     *  convention every weight consumer expects. GGUF stores ggml dims
     *  `[ne0=in, ne1=out, ...]` with bytes row-major over the last dim, so
     *  `[dims[last], product(earlier dims)]` reinterprets the same bytes
     *  correctly. 1-D tensors (norms, biases) are returned unchanged —
     *  collapsing them to `[d,1]` would break the 1-D broadcast in
     *  LayerNorm/RMSNorm and bias adds. Mirrors decodeQ4KM/Q6K's rows/cols. */
    private static function orientDims(dims:Array<Int>):Array<Int> {
        if (dims.length <= 1) return dims;
        var rows = dims[dims.length - 1];
        var cols = 1;
        for (i in 0...dims.length - 1) cols *= dims[i];
        return [rows, cols];
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

    /** Convert a Q5_0 (GGUF dtype 6) tensor into an INT8 per-row QTensor.
     *  Unlike the k-quant decoders this is not zero-copy — each row is
     *  dequantised once and re-encoded (int8 per-row keeps at least Q5_0's
     *  16-level granularity unless one row's block scales span >8×) — but
     *  the result stays quantised and rides the integer-dot matmul path. */
    private static function decodeQ5_0Int8(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):QTensor {
        var rows = info.dims[info.dims.length - 1];
        var cols = 1;
        for (i in 0...info.dims.length - 1) cols *= info.dims[i];
        var qt = QTensor.fromBytesQ5_0Int8(raw, rows, cols);
        if (qt == null) {
            throw "GGUFLoader: QTensor.fromBytesQ5_0Int8 returned null for '"
                + info.name + "' (rows=" + rows + ", cols=" + cols + ").";
        }
        return qt;
    }

    /** Convert a Q5_1 (GGUF dtype 7) tensor into an INT8 per-row QTensor.
     *  Q5_1 is Q5_0 plus an explicit per-block min (`y = d*q + m`); both are
     *  32-element legacy blocks, so both land on the INT8 integer-dot path. */
    private static function decodeQ5_1Int8(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):QTensor {
        var rows = info.dims[info.dims.length - 1];
        var cols = 1;
        for (i in 0...info.dims.length - 1) cols *= info.dims[i];
        var qt = QTensor.fromBytesQ5_1Int8(raw, rows, cols);
        if (qt == null) {
            throw "GGUFLoader: QTensor.fromBytesQ5_1Int8 returned null for '"
                + info.name + "' (rows=" + rows + ", cols=" + cols + ").";
        }
        return qt;
    }

    /** Convert a Q5_K (GGUF dtype 13) tensor into a Q4_K_M QTensor. Not
     *  zero-copy — each super-block is dequantised and re-encoded once at
     *  load — but it preserves Q5_K's per-32-element scale+min structure,
     *  rides the fastest matmul kernel, and ends up SMALLER than the source
     *  (4.5 vs 5.5 bits/weight). */
    private static function decodeQ5KToQ4KM(raw:haxe.io.Bytes, info:GGUFReader.TensorInfo):QTensor {
        var rows = info.dims[info.dims.length - 1];
        var cols = 1;
        for (i in 0...info.dims.length - 1) cols *= info.dims[i];
        var qt = QTensor.fromBytesQ5KQ4KM(raw, rows, cols);
        if (qt == null) {
            throw "GGUFLoader: QTensor.fromBytesQ5KQ4KM returned null for '"
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
    var tokenizer:Tokenizer;
    var metadata:ModelMetadata;
};
