// Regression test: SIMD4i32 (the integer i32x4 companion to SIMD4f) must work
// end-to-end on BOTH native and wasm.
//
// This exercises two fixes that were jointly required:
//
// 1. Element type on VectorExtract/Insert/Reduce. These IR instructions
//    previously carried no element type; the wasm backend inferred it from
//    register_types, which is lost when a vector wrapper is inlined — so an
//    inlined integer reduce emitted f32x4.extract_lane on an i32 vector
//    (reading i32 lanes as floats → 0). Now elem_ty travels with the
//    instruction and survives inlining.
//
// 2. SIMD instance-method dispatch. A vector receiver was hard-coded to the
//    "rayzor_SIMD4f" class hint, so SIMD4i32.sum()/get() dispatched to the
//    SIMD4f (f32) wrappers. Native masked this (Cranelift reduces/extracts by
//    SSA value type, which is i32x4); wasm exposed it. Now the hint
//    distinguishes i32x4 → SIMD4i32 from f32x4 → SIMD4f.
//
// Native worked by accident even when broken, so the *values* must be checked,
// and a wasm run is the real gate.

import rayzor.SIMD4i32;

class Main {
    static function main() {
        var a = SIMD4i32.make(1, 2, 3, 4);
        var b = SIMD4i32.splat(10);

        var add = a + b; // [11, 12, 13, 14]
        var mul = a * b; // [10, 20, 30, 40]
        var sub = b - a; // [9, 8, 7, 6]

        // Constant lanes.
        var l0 = add.get(0);
        var l3 = add.get(3);
        // Runtime lane (loop var) — exercises the select-chain extract.
        var loopSum = 0;
        for (i in 0...4) loopSum += mul.get(i);

        var ok = l0 == 11 && l3 == 14
            && add.sum() == 50 && mul.sum() == 100 && sub.sum() == 30
            && loopSum == 100;

        if (ok) {
            Sys.println("PASS simd4i32 add.sum=" + add.sum() + " mul.sum=" + mul.sum()
                + " lanes=" + l0 + "," + l3);
        } else {
            Sys.println("FAIL add.sum=" + add.sum() + " mul.sum=" + mul.sum()
                + " sub.sum=" + sub.sum() + " l0=" + l0 + " l3=" + l3 + " loopSum=" + loopSum);
        }
    }
}
