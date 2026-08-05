// 256-bit SIMD regression test: SIMD8i32/SIMD32i8 must agree exactly with the
// 128-bit pair they replace. The whole point of the wide types is that one
// 32-byte dot equals two 16-byte dots, so a width bug shows up as a mismatch
// here rather than as slightly-wrong logits three layers deep.
//
// LLVM-only: wasm's v128 and Cranelift have no 256-bit vector and refuse these
// types, so the wide half is gated. The 128-bit reference runs everywhere.

import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
#if llvm
import rayzor.SIMD8i32;
import rayzor.SIMD32i8;
#end

class Main {
    static function main() {
        var z4 = SIMD4i32.splat(0);

        // 128-bit reference: two dots over the same 32 bytes the wide type sees.
        var refSigned = SIMD4i32.dot(z4, SIMD16i8.splat(2), SIMD16i8.splat(-3)).sum()
            + SIMD4i32.dot(z4, SIMD16i8.splat(2), SIMD16i8.splat(-3)).sum();
        var refI7 = SIMD4i32.dotI8I7(z4, SIMD16i8.splat(-2), SIMD16i8.splat(15)).sum()
            + SIMD4i32.dotI8I7(z4, SIMD16i8.splat(-2), SIMD16i8.splat(15)).sum();
        var refU8 = SIMD4i32.dotI8U8(z4, SIMD16i8.splat(-2), SIMD16i8.splat(143)).sum()
            + SIMD4i32.dotI8U8(z4, SIMD16i8.splat(-2), SIMD16i8.splat(143)).sum();

        #if llvm
        var z8 = SIMD8i32.splat(0);
        var wideSigned = SIMD8i32.dot(z8, SIMD32i8.splat(2), SIMD32i8.splat(-3)).sum();
        var wideI7 = SIMD8i32.dotI8I7(z8, SIMD32i8.splat(-2), SIMD32i8.splat(15)).sum();
        var wideU8 = SIMD8i32.dotI8U8(z8, SIMD32i8.splat(-2), SIMD32i8.splat(143)).sum();

        // Nibble unpack on the wide type, the Q4 kernel's actual use.
        var packed = SIMD32i8.splat(0x53);
        var lo = SIMD32i8.and(packed, SIMD32i8.splat(0x0F));
        var hi = SIMD32i8.ushr(packed, 4);
        var loDot = SIMD8i32.dotI8I7(z8, SIMD32i8.splat(1), lo).sum();
        var hiDot = SIMD8i32.dotI8I7(z8, SIMD32i8.splat(1), hi).sum();

        var ok = wideSigned == refSigned && wideI7 == refI7 && wideU8 == refU8
            && wideSigned == -192 && wideI7 == -960 && wideU8 == -9152
            && loDot == 32 * 3 && hiDot == 32 * 5;
        if (ok) {
            Sys.println("PASS simd256-dot signed=" + wideSigned + " i7=" + wideI7
                + " u8=" + wideU8 + " lo=" + loDot + " hi=" + hiDot);
        } else {
            Sys.println("FAIL simd256-dot signed=" + wideSigned + "/" + refSigned
                + " i7=" + wideI7 + "/" + refI7 + " u8=" + wideU8 + "/" + refU8
                + " lo=" + loDot + " hi=" + hiDot);
        }
        #else
        if (refSigned == -192 && refI7 == -960 && refU8 == -9152) {
            Sys.println("PASS simd256-dot (128-bit reference only, no 256-bit on this target)");
        } else {
            Sys.println("FAIL simd256-dot reference signed=" + refSigned
                + " i7=" + refI7 + " u8=" + refU8);
        }
        #end
    }
}
