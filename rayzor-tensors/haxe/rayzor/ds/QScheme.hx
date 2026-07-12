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
 * - `Q6_K`: GGUF's 6-bit K-quant. 256-element super-blocks, 210 bytes,
 *   i8 per-sub-block scales and an f16 super-block scale. Q4_K_M GGUFs
 *   promote accuracy-sensitive weights (e.g. `attn_v`, `ffn_down`) to Q6_K.
 *
 * Enum order is load-bearing — ordinals match the runtime tag values
 * (`QSCHEME_INT8 = 0`, `QSCHEME_Q4_K_M = 1`, `QSCHEME_Q6_K = 2`).
 */
enum QScheme {
    INT8;
    Q4_K_M;
    Q6_K;
}
