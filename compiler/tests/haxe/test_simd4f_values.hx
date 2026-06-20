// Regression test: SIMD4f must produce CORRECT VALUES, not merely run.
//
// Two bugs made SIMD4f compile/execute while yielding garbage:
//   1. `a + b` resolved its @:op to `Usize.add` (integer add) instead of the
//      inline VectorBinOp path, emitting `iadd.f32` (trap) — fixed in
//      tast_to_hir (skip @:op rewrite for SIMD4f arithmetic).
//   2. `splat`/`make` pass Haxe `Float` (f64) args to F32-lane wrapper params
//      with no demote, so the f64 bit-pattern landed in the f32 lanes as
//      garbage (splat(4.0).sqrt() summed to nonsense) — fixed in hir_to_mir
//      (coerce mir-wrapper static-call args to declared param types).
//
// test_simd_e2e only checks "execution succeeded", so it passed on garbage.
// This asserts values via HORIZONTAL ops (sum/dot exercise every lane), which
// avoids the separate per-lane `.get(lane)` extraction bug.
import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
        var b = SIMD4f.make(5.0, 6.0, 7.0, 8.0);

        var addSum = (a + b).sum();   // 6+8+10+12 = 36
        var mulSum = (a * b).sum();   // 5+12+21+32 = 70
        var dot = a.dot(b);           // 70
        var spSum = SIMD4f.splat(4.0).sum();        // 16
        var rtSum = SIMD4f.splat(4.0).sqrt().sum(); // 2+2+2+2 = 8

        var ok = addSum == 36.0 && mulSum == 70.0 && dot == 70.0
            && spSum == 16.0 && rtSum == 8.0;

        if (ok) {
            Sys.println("PASS simd4f-values add=" + addSum + " mul=" + mulSum
                + " dot=" + dot + " splat=" + spSum + " sqrt=" + rtSum);
        } else {
            Sys.println("FAIL add=" + addSum + " mul=" + mulSum + " dot=" + dot
                + " splat=" + spSum + " sqrt=" + rtSum);
        }
    }
}
