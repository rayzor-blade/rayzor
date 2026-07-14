package nue.transformer;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * Attention module — a `Module` that additionally accepts a padding mask.
 *
 * `TransformerBlock` holds its attention sublayer as this type (not the
 * bare `Module`) so the masked block path can call `forwardMasked` through
 * ordinary virtual dispatch. Reaching a concrete attention type by
 * downcasting a `Module` field is what this interface exists to avoid.
 *
 * `forwardMasked` adds a caller-supplied additive bias to the attention
 * scores before softmax — a `[seq_key]` vector, `0` for real keys and a
 * large negative for padded keys — so batched/padded inputs ignore
 * padding. Non-padding callers use plain `forward`.
 */
interface Attention extends Module {
    function forwardMasked(x:Tensor, attnBias:Tensor):Tensor;
}
