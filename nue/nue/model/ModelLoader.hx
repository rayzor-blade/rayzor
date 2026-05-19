package nue.model;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * Format-agnostic interface for loading a model from disk.
 *
 * Pipeline (every concrete loader follows this):
 * ```
 *   readMetadata(path)        → ModelMetadata
 *   readNamedTensors(path)    → Map<String, Tensor>
 *   nue.arch.ArchRegistry.build(metadata, weights) → Module
 * ```
 *
 * The loader's job is parsing the on-disk format. The architecture
 * selection (instantiating the right Module tree for "llama" vs "gpt2"
 * etc.) is delegated to `nue.arch.ArchRegistry` so the same loader
 * works across architectures — drop in a new arch builder, and every
 * format that names it gains support.
 *
 * `tieWordEmbeddings` requires post-processing: when true, the LM
 * head's weight tensor is the embed table itself. The arch builder
 * handles that rebind so loaders don't have to know.
 */
interface ModelLoader {
    /** Parse top-level metadata without decoding tensor data. */
    function readMetadata(path:String):ModelMetadata;

    /**
     * Decode every weight tensor into a `Tensor` (dtype determined by
     * the file format — F32, F16, or quantised via QTensor). Keys are
     * the canonical GGUF naming (`token_embd.weight`,
     * `blk.0.attn_q.weight`, …) regardless of file format — the loader
     * normalises ONNX/safetensors names into this convention so the
     * arch builder doesn't need to know which format the bytes came from.
     */
    function readNamedTensors(path:String):Map<String, Tensor>;

    /**
     * Convenience wrapper: read metadata + tensors, look up the
     * architecture builder, and return a ready-to-call Module.
     */
    function load(path:String):Module;
}
