// Lane-wise convert and signed-saturating pack: the ops that let a quantise
// produce its bytes without leaving the vector domain.
//
// Both must mean the same thing on every tier, so the interesting cases are
// the edges: out-of-range floats (saturate, never wrap or trap), NaN (-> 0),
// negative truncation direction, and lanes past ±127 in the pack.
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Mem;
import rayzor.Usize;
import rayzor.Bytes;

class Main {
    static function main():Void {
        // f32 -> i32 truncates toward zero.
        var t = SIMD4i32.fromFloat(SIMD4f.make(1.9, -1.9, 0.4, -0.4));
        eq("trunc", [t.get(0), t.get(1), t.get(2), t.get(3)], [1, -1, 0, 0]);

        // Round-to-nearest-even first, then convert.
        var r = SIMD4i32.fromFloat(SIMD4f.make(0.5, 1.5, 2.5, -2.5).round());
        eq("round", [r.get(0), r.get(1), r.get(2), r.get(3)], [0, 2, 2, -2]);

        // i32 -> f32 and back is the identity for small ints.
        var back = SIMD4i32.fromFloat(SIMD4f.fromInt(SIMD4i32.make(-7, 0, 13, 99)));
        eq("roundtrip", [back.get(0), back.get(1), back.get(2), back.get(3)], [-7, 0, 13, 99]);

        // The pack saturates rather than wrapping.
        var p = SIMD16i8.packI32(
            SIMD4i32.make(0, 1, -1, 127),
            SIMD4i32.make(-128, 128, -129, 1000),
            SIMD4i32.make(-1000, 7, -7, 42),
            SIMD4i32.make(126, -126, 5, -5));
        eq("pack", [p.get(0), p.get(1), p.get(2), p.get(3)], [0, 1, -1, 127]);
        eq("pack.sat", [p.get(4), p.get(5), p.get(6), p.get(7)], [-128, 127, -128, 127]);
        eq("pack.c", [p.get(8), p.get(9), p.get(10), p.get(11)], [-128, 7, -7, 42]);
        eq("pack.d", [p.get(12), p.get(13), p.get(14), p.get(15)], [126, -126, 5, -5]);

        // The whole produce side of a quantise, end to end and written out.
        var b = Bytes.alloc(64);
        var base:Usize = b.address();
        var lanes = [3.4, -3.6, 200.0, -200.0];
        var v = SIMD4f.make(lanes[0], lanes[1], lanes[2], lanes[3]);
        var q = SIMD4i32.fromFloat(v.round());
        SIMD16i8.packI32(q, q, q, q).store(Ptr.fromRaw(base));
        eq("store", [Mem.loadU8(base), Mem.loadU8(base + Usize.fromInt(1)),
                     Mem.loadU8(base + Usize.fromInt(2)), Mem.loadU8(base + Usize.fromInt(3))],
                    [3, 252, 127, 128]);   // 3, -4, +127 sat, -128 sat, as unsigned bytes

        Sys.println("simd convert/pack ok");
    }

    static function eq(tag:String, got:Array<Int>, want:Array<Int>):Void {
        for (i in 0...want.length) {
            if (got[i] != want[i]) {
                Sys.println("FAIL " + tag + "[" + i + "]: got " + got[i] + " want " + want[i]);
            }
        }
    }
}
