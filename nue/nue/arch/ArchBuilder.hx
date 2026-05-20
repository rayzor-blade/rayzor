package nue.arch;

import nue.Module;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import rayzor.ds.Tensor;

/**
 * Interface for an architecture builder.
 *
 * Each supported model family (Llama, GPT-2, Mistral, Falcon, Phi, …)
 * provides an `ArchBuilder` that knows how to wire its specific
 * `Module` tree out of nue's generic primitives (`Embedding`, `Linear`,
 * `RMSNorm` / `LayerNorm`, `GQAttention` / `MultiHeadAttention`,
 * `SwiGLU` / `GeluFFN`, etc.) given a `ModelMetadata` shape spec and a
 * dictionary of named weight tensors.
 *
 * Format loaders dispatch through `ArchRegistry.build(metadata, weights)`,
 * so adding an architecture is a single registry entry plus this
 * implementation — every file format that names it gains support.
 *
 * `validate(metadata)` lets the builder reject configs it doesn't
 * handle (e.g. an MoE-only builder refusing a non-MoE config) BEFORE
 * weight loading starts.
 */
interface ArchBuilder {
    /** Lowercase family name as used by file metadata (`"llama"`, `"gpt2"`). */
    function name():String;

    /** Quick sanity check on the metadata's shape fields. */
    function validate(metadata:ModelMetadata):Bool;

    /** Wire the Module tree using the supplied named weight tensors. */
    function build(metadata:ModelMetadata, weights:NamedTensorMap):Module;
}
