package nue;

import rayzor.ds.Tensor;

/**
 * Common interface for every nue building block — RMSNorm, RoPE, Linear,
 * Embedding, attention blocks, full transformer stacks.
 *
 * A Module owns its parameters (weight tensors) and exposes a single
 * `forward(x)` step. `parameters()` is intentionally a flat list keyed by
 * name so the GGUF loader (Phase 7) can map a tensor file's named entries
 * straight onto the module tree without needing reflection over field
 * structures.
 *
 * Lightweight by design — no autograd, no graph capture. Inference only.
 * If we later add backprop, the Module surface grows a `backward(grad)`
 * but the forward path stays unchanged.
 *
 * Example shape:
 * ```haxe
 * class MyLinear implements Module {
 *     var weight:Tensor;
 *     public function new(w:Tensor) { this.weight = w; }
 *     public function forward(x:Tensor):Tensor return x.matmul(weight);
 *     public function parameters():Array<NamedTensor> return [{ name: "weight", tensor: weight }];
 * }
 * ```
 */
interface Module {
    /** Run a single forward pass. */
    function forward(x:Tensor):Tensor;

    /**
     * Flat list of named weight tensors owned by this module (recursing into
     * sub-modules). Used by the loader to populate weights from GGUF.
     *
     * Names should match the GGUF tensor naming convention for the model
     * family being implemented — e.g. for Llama:
     *   "blk.{layer}.attn_q.weight", "blk.{layer}.attn_norm.weight", ...
     */
    function parameters():Array<NamedTensor>;
}

/**
 * Tagged tensor used by `Module.parameters()` so the loader can address
 * weights by name without depending on Haxe reflection.
 */
typedef NamedTensor = {
    var name:String;
    var tensor:Tensor;
}
