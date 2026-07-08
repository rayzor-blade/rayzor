// Regression test: SIMD4i32.dot — the fused widening integer dot-product
// (the quantized-matmul primitive). 16 i8×i8 products of a,b summed in groups
// of 4 into the 4 i32 lanes of an accumulator.
//
// Lowers to a single AArch64 SDOT (vdotq_s32) on native via the
// swiden+imul+iadd_pairwise CLIF tree, to i32x4.relaxed_dot_i8x16_i7x16_add_s
// on wasm (→ SDOT under wasmtime 47), and to the sext→mul→partial-reduce idiom
// on the LLVM tier (→ SDOT). All four paths must agree on the values; values
// (not just exit codes) are checked, and a wasm run is the real gate.

import rayzor.SIMD4i32;
import rayzor.SIMD16i8;

class Main {
    static function main() {
        var a = SIMD16i8.splat(2);
        var b = SIMD16i8.splat(3);

        // dot(0, splat(2), splat(3)): each i32 lane = 4 products × (2·3) = 24.
        var acc = SIMD4i32.dot(SIMD4i32.splat(0), a, b);
        var sum1 = acc.sum(); // 4 × 24 = 96
        var lane0 = acc.get(0); // 24

        // Accumulate into the same vector again → each lane 48.
        acc = SIMD4i32.dot(acc, a, b);
        var sum2 = acc.sum(); // 192

        // Different i7-range values (deterministic across runtimes).
        var sum3 = SIMD4i32.dot(SIMD4i32.splat(0), SIMD16i8.splat(5), SIMD16i8.splat(7)).sum(); // 4×4×35 = 560

        // Signedness regression: x86 VPDPBUSD is unsigned*signed, but the
        // public SIMD4i32.dot contract is signed*signed. Q8 flash attention
        // depends on negative lanes in both operands staying signed.
        var sum4 = SIMD4i32.dot(SIMD4i32.splat(0), SIMD16i8.splat(2), SIMD16i8.splat(-3)).sum(); // -96
        var sum5 = SIMD4i32.dot(SIMD4i32.splat(0), SIMD16i8.splat(-2), SIMD16i8.splat(3)).sum(); // -96

        var ok = sum1 == 96 && lane0 == 24 && sum2 == 192 && sum3 == 560
            && sum4 == -96 && sum5 == -96;
        if (ok) {
            Sys.println("PASS simd-dot sum1=" + sum1 + " sum2=" + sum2
                + " sum3=" + sum3 + " sum4=" + sum4 + " sum5=" + sum5);
        } else {
            Sys.println("FAIL sum1=" + sum1 + " lane0=" + lane0 + " sum2=" + sum2
                + " sum3=" + sum3 + " sum4=" + sum4 + " sum5=" + sum5);
        }
    }
}
