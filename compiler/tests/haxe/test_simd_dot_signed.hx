// x86 regression test: SIMD4i32.dot must be signed i8 x signed i8.
// AVX-VNNI's VPDPBUSD is unsigned*signed, so using it for the public
// SIMD4i32.dot primitive corrupts Q8 x Q8 attention when either side has
// negative lanes. Keep this as a separate canary so stale MIR for
// test_simd_dot.hx cannot hide the signedness check.

import rayzor.SIMD4i32;
import rayzor.SIMD16i8;

class Main {
    static function main() {
        var z = SIMD4i32.splat(0);

        var posNeg = SIMD4i32.dot(z, SIMD16i8.splat(2), SIMD16i8.splat(-3)).sum();
        var negPos = SIMD4i32.dot(z, SIMD16i8.splat(-2), SIMD16i8.splat(3)).sum();
        var negNeg = SIMD4i32.dot(z, SIMD16i8.splat(-2), SIMD16i8.splat(-3)).sum();
        var i8i7 = SIMD4i32.dotI8I7(z, SIMD16i8.splat(-2), SIMD16i8.splat(15)).sum();
        var i8u8 = SIMD4i32.dotI8U8(z, SIMD16i8.splat(-2), SIMD16i8.splat(143)).sum();

        var ok = posNeg == -96 && negPos == -96 && negNeg == 96
            && i8i7 == -480 && i8u8 == -4576;
        if (ok) {
            Sys.println("PASS simd-dot-signed posNeg=" + posNeg
                + " negPos=" + negPos + " negNeg=" + negNeg
                + " i8i7=" + i8i7 + " i8u8=" + i8u8);
        } else {
            Sys.println("FAIL simd-dot-signed posNeg=" + posNeg
                + " negPos=" + negPos + " negNeg=" + negNeg
                + " i8i7=" + i8i7 + " i8u8=" + i8u8);
        }
    }
}
