// Regression test: SIMD16i8.load reading directly out of a rayzor.Bytes buffer
// via Bytes.address() + Ptr.fromRaw. This is the data-access path the pure-Haxe
// Q4_K_M dot kernel needs to read packed weight bytes (Phase 3 of the VectorDot
// plan). Verifies: address() returns a usable base, byte-granular offset
// arithmetic (base + 16), and that load reads the correct 16 bytes IN ORDER.
//
// NATIVE only: on the wasm host-import path the Bytes buffer is not in guest
// linear memory, so a guest V128Load against this address is invalid. The
// in-guest wasm kernel reads weights through the QTensor data pointer instead.
//
// Read-out uses the already-verified dot+sum primitives. Order is checked by
// dotting two DIFFERENT loaded regions (a ramp × a ramp is order-sensitive).

import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Bytes;

class Main {
    static function main() {
        // 32 bytes: region A [0..16) = i+1 (1..16); region B [16..32) = i (0..15).
        var buf = Bytes.alloc(32);
        for (i in 0...16) {
            buf.set(i, i + 1);       // A: 1,2,...,16
            buf.set(16 + i, i);      // B: 0,1,...,15  (i7 range → valid dot operand)
        }
        // Fresh address() per load — Usize is a value type and reusing one
        // `var` across both loads trips the known false-ownership warning.
        var aVec = SIMD16i8.load(Ptr.fromRaw(buf.address()));                       // A
        var bVec = SIMD16i8.load(Ptr.fromRaw(buf.address() + Usize.fromInt(16)));   // B (byte offset 16)

        var one = SIMD16i8.splat(1);
        var z = SIMD4i32.splat(0);

        // Σ of region A = 1+2+...+16 = 136 (verifies load + base address).
        var sumA = SIMD4i32.dot(z, aVec, one).sum();
        // Σ A[i]*B[i] = Σ_{i=0..15} (i+1)*i = 1240 + 120 = 1360
        // (order-sensitive: a reversed load would not give 1360 → verifies
        //  byte order within the load AND the +16 byte-offset addressing).
        var ordered = SIMD4i32.dot(z, aVec, bVec).sum();

        var ok = sumA == 136 && ordered == 1360;
        if (ok) {
            Sys.println("PASS simd16i8-load-bytes sumA=" + sumA + " ordered=" + ordered);
        } else {
            Sys.println("FAIL sumA=" + sumA + " ordered=" + ordered);
        }
    }
}
