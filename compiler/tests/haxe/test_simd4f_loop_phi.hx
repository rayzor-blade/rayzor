// Regression test: a SIMD vector used as a loop-carried accumulator (a
// vector-typed phi) must compile and produce correct values.
//
// Before the fix: the Cranelift backend's phi block-param type match had no
// `Vector` arm, so a `<4 x f32>` phi (a SIMD4f loop accumulator) fell to the
// `_ => I64` default. The block param became a GPR while the incoming values
// were FPR/vector regs, and AArch64 regalloc panicked in `gen_move` with
// `assertion failed: to_reg.class() == from_reg.class()` (W0020, the function
// couldn't be defined). This blocked every banded SIMD reduction kernel
// (RMSNorm sum-of-squares, dot products) — exactly the pure-Haxe nue kernel
// shape: load -> fma into a SIMD accumulator across a loop -> horizontal sum.
//
// After the fix: vector phis get the correct vector Cranelift type in both
// the block-param creation and the phi-arg coercion, so block param and
// incoming values share the vector reg class.

import rayzor.SIMD4f;

class Main {
    static function main() {
        // Vector accumulator carried across a loop (the phi under test).
        var acc = SIMD4f.splat(0.0);
        var i = 0;
        while (i < 100) {
            acc = acc + SIMD4f.splat(2.0);
            i++;
        }
        // acc = [200, 200, 200, 200]; horizontal sum = 800.
        var total = acc.sum();

        // Also a mul-accumulate variant (the FMA shape kernels use).
        var acc2 = SIMD4f.splat(1.0);
        var j = 0;
        while (j < 4) {
            acc2 = acc2 * SIMD4f.splat(2.0);
            j++;
        }
        // acc2 = [16, 16, 16, 16]; sum = 64.
        var total2 = acc2.sum();

        if (total == 800.0 && total2 == 64.0) {
            Sys.println("PASS simd4f-loop-phi add=" + total + " mul=" + total2);
        } else {
            Sys.println("FAIL add=" + total + " (want 800) mul=" + total2 + " (want 64)");
        }
    }
}
