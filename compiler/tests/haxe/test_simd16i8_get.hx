// Regression test: SIMD16i8.get(lane) — read one i8 lane (sign-extended to Int)
// out of a loaded vector, in-guest (no Ptr.deref, no per-byte Bytes FFI). This
// is how the pure-Haxe Q4 dot kernel pulls the block header (d/dmin/scales) out
// of the first 16 weight bytes. Bytes are signed i8; mask `& 0xFF` for unsigned.
//
// Values checked native + wasm (wasm is the real gate — it lowers via
// I8x16ExtractLaneS, which sign-extends to i32 exactly like the native path).

import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Bytes;

class Main {
    static function main() {
        var b = Bytes.alloc(16);
        for (i in 0...16) b.set(i, i * 16 + 7); // 7, 23, 39, ... 247 (top ones > 127)
        var v = SIMD16i8.load(Ptr.fromRaw(b.address()));

        // Unsigned read-back (& 0xFF) of a few lanes.
        var l0:Int = v.get(0) & 0xFF;
        var l1:Int = v.get(1) & 0xFF;
        var l5:Int = v.get(5) & 0xFF;
        var l15:Int = v.get(15) & 0xFF;
        // Signed lane 8 = 135 as u8 = -121 as i8 (sign-extend check).
        var l8signed:Int = v.get(8);

        var ok = l0 == 7 && l1 == 23 && l5 == 87 && l15 == 247 && l8signed == -121;
        if (ok) {
            Sys.println("PASS simd16i8-get l0=" + l0 + " l1=" + l1 + " l5=" + l5
                + " l15=" + l15 + " l8signed=" + l8signed);
        } else {
            Sys.println("FAIL l0=" + l0 + " l1=" + l1 + " l5=" + l5
                + " l15=" + l15 + " l8signed=" + l8signed);
        }
    }
}
