package rayzor.ds;

/**
 * Quantisation scheme tag for `QTensor`.
 *
 * - `INT8`: 8-bit symmetric per-row quant. Each row of the matrix carries
 *   one f32 scale. 4× memory reduction vs F32; ~1% accuracy loss on
 *   typical LLM weights.
 *
 * - `Q4_K_M`: GGUF's K-quant format. 256-element super-blocks, 4-bit
 *   weights, 6-bit per-sub-block (scale, min) pairs. The standard format
 *   for shipping Llama-class models to edge. ~5-bit-equivalent storage
 *   per weight, near-lossless for most prompts.
 *
 * Enum order is load-bearing — ordinals match the runtime tag values
 * (`QSCHEME_INT8 = 0`, `QSCHEME_Q4_K_M = 1`).
 */
enum QScheme {
    INT8;
    Q4_K_M;
}
