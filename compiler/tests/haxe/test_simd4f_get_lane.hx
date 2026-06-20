// Regression test: SIMD4f.get(lane) must return the value of the requested
// lane, for both constant and runtime lane indices.
//
// Before the fix: the SIMD4f_extract MIR wrapper hard-coded
// `vector_extract(self, 0)`, ignoring its `lane` parameter, so `v.get(i)`
// always returned lane 0 regardless of `i` (e.g. make(10,20,30,40).get(2)
// gave 10). VectorExtract requires a compile-time-constant lane, so the fix
// extracts all four lanes with constant indices and selects the one matching
// the runtime `lane` via a branchless Select chain. That also required adding
// a `Select` arm to the Cranelift backend (normal Haxe ternaries lower to
// branches+phi, so Select had never been emitted on the native tier before —
// it would otherwise W0020-trap).
//
// This gives pure-Haxe code a working scalar f32 read: `SIMD4f.load(p).get(i)`
// reads lane `i` of an f32 buffer without going through per-element FFI.

import rayzor.SIMD4f;

class Main {
    static function main() {
        var v = SIMD4f.make(10.0, 20.0, 30.0, 40.0);

        // Constant lane indices.
        var c0 = v.get(0);
        var c1 = v.get(1);
        var c2 = v.get(2);
        var c3 = v.get(3);

        // Runtime lane index (loop variable) — exercises the Select chain.
        var sum = 0.0;
        for (i in 0...4) sum += v.get(i);

        // sum() is the known-good horizontal-add cross-check.
        var hsum = v.sum();

        var ok = c0 == 10.0 && c1 == 20.0 && c2 == 30.0 && c3 == 40.0
            && sum == 100.0 && hsum == 100.0;
        if (ok) {
            Sys.println("PASS simd4f-get-lane lanes=" + c0 + "," + c1 + "," + c2 + "," + c3
                + " loopSum=" + sum);
        } else {
            Sys.println("FAIL lanes=" + c0 + "," + c1 + "," + c2 + "," + c3
                + " loopSum=" + sum + " hsum=" + hsum);
        }
    }
}
