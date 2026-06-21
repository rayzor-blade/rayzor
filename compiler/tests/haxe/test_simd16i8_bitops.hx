// Regression test: SIMD16i8 lane-wise bitwise (and/or/xor) + shift (shl/shr/ushr).
// These are the Q4_K_M nibble-unpack primitives: low nibble = AND 0x0F, high
// nibble = USHR 4. The wasm backend already lowered i8x16 And/Shl/ShrS/ShrU;
// this test is the gate that the NATIVE (Cranelift) backend now lowers the
// shifts too (ishl/sshr/ushr) and that both backends agree on the values.
//
// Read-out uses only the already-verified dot+sum primitives: an i8x16 whose 16
// lanes are all equal sums to 16×lane under dot(.,v,splat(1)) (b=1 is in i7
// range → cross-runtime deterministic). The dot is inlined per case (passing a
// SIMD16i8 across a function boundary is a separate codegen path). Values (not
// just exit codes) are checked, and a wasm run is the real gate.

import rayzor.SIMD4i32;
import rayzor.SIMD16i8;

class Main {
    static function main() {
        // 0xA7 = 167; the low byte 0xA7 is -89 as a signed i8 (top bit set),
        // which distinguishes arithmetic (shr) from logical (ushr) right shift.
        var one = SIMD16i8.splat(1);
        var zero = SIMD4i32.splat(0);

        // and 0x0F → low nibble 0x07 = 7 → 16×7 = 112
        var andLow = SIMD4i32.dot(zero, SIMD16i8.and(SIMD16i8.splat(0xA7), SIMD16i8.splat(0x0F)), one).sum();
        // ushr 4 → high nibble 0x0A = 10 → 160
        var hiNib = SIMD4i32.dot(zero, SIMD16i8.ushr(SIMD16i8.splat(0xA7), 4), one).sum();
        // shr 4 (arithmetic) → -89>>4 = -6 → -96
        var arShr = SIMD4i32.dot(zero, SIMD16i8.shr(SIMD16i8.splat(0xA7), 4), one).sum();
        // shl 4 → 0xA7<<4 low byte = 0x70 = 112 → 1792
        var shl4 = SIMD4i32.dot(zero, SIMD16i8.shl(SIMD16i8.splat(0xA7), 4), one).sum();
        // or  0x05|0x02 = 0x07 = 7 → 112
        var orV = SIMD4i32.dot(zero, SIMD16i8.or(SIMD16i8.splat(0x05), SIMD16i8.splat(0x02)), one).sum();
        // xor 0x0F^0x09 = 0x06 = 6 → 96
        var xorV = SIMD4i32.dot(zero, SIMD16i8.xor(SIMD16i8.splat(0x0F), SIMD16i8.splat(0x09)), one).sum();

        var ok = andLow == 112 && hiNib == 160 && arShr == -96
              && shl4 == 1792 && orV == 112 && xorV == 96;
        if (ok) {
            Sys.println("PASS simd16i8-bitops and=" + andLow + " ushr=" + hiNib
                + " shr=" + arShr + " shl=" + shl4 + " or=" + orV + " xor=" + xorV);
        } else {
            Sys.println("FAIL and=" + andLow + " ushr=" + hiNib + " shr=" + arShr
                + " shl=" + shl4 + " or=" + orV + " xor=" + xorV);
        }
    }
}
