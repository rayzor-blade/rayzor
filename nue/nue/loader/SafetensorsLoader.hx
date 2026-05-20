package nue.loader;

import nue.Module;
import nue.model.ModelLoader;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import rayzor.ds.Tensor;

/**
 * Safetensors (`*.safetensors`) loader — Hugging Face's preferred
 * format for sharing trained weights. Layout is dead-simple:
 *
 * ```
 *   u64 header_len             (little-endian)
 *   header_len bytes of JSON   (per-tensor names, shapes, dtypes,
 *                               byte ranges into the data section)
 *   tensor data                (raw bytes, contiguous, aligned)
 * ```
 *
 * Compared to GGUF: no quantisation built in (everything is plain
 * F32 / F16 / BF16), no embedded tokenizer (callers grab the
 * `tokenizer.json` sidecar separately), and no architecture hint
 * (loader has to infer from companion `config.json` or accept it
 * from the caller).
 *
 * **Status:** stub — the loader surface is in place, parsing is not.
 * Wire up against:
 *   1. A streaming JSON parser (or just `haxe.Json.parse(headerStr)`).
 *   2. The per-tensor `data_offsets: [start, end]` ranges into the
 *      data section.
 *   3. Per-architecture name mapping (HF naming differs from GGUF:
 *      `model.embed_tokens.weight` instead of `token_embd.weight`, etc.).
 *
 * The renaming step is the substantive work — every architecture
 * builder consumes GGUF-style names, so we keep them as the canonical
 * convention and translate at load time.
 */
class SafetensorsLoader implements ModelLoader {
    public function new() {}

    public function readMetadata(_:String):ModelMetadata {
        throw "SafetensorsLoader: not yet implemented. Use GGUF for now, "
            + "or wire HF config.json + safetensors header parsing here.";
    }

    public function readNamedTensors(_:String):NamedTensorMap {
        throw "SafetensorsLoader: not yet implemented.";
    }

    public function load(_:String):Module {
        throw "SafetensorsLoader: not yet implemented.";
    }
}
