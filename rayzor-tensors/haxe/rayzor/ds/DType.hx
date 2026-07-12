package rayzor.ds;

/**
 * Data type for tensor elements.
 *
 * Determines the numeric precision and storage size of each element
 * in a Tensor. Maps to runtime dtype tags (i64 constants) at MIR level.
 *
 * Enum order is load-bearing — the ordinals 0..N are the runtime
 * tag values consumed by `runtime/src/tensor.rs` and the GPU side
 * at `gpu/src/buffer.rs`. Do not reorder; append new variants only.
 *
 * - F32: 32-bit float (default, broadest kernel coverage)
 * - F16: IEEE 754 half (2-byte storage, common dequant target)
 * - BF16: brain float (2-byte storage, larger range than F16)
 * - I32 / I8 / U8: integer formats
 * - FP8_E4M3: 8-bit float, 4-bit exponent / 3-bit mantissa
 * - FP8_E5M2: 8-bit float, 5-bit exponent / 2-bit mantissa
 */
enum DType {
    F32;
    F16;
    BF16;
    I32;
    I8;
    U8;
    FP8_E4M3;
    FP8_E5M2;
}
