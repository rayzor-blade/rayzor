// The produce side of an int8 quantise written in the vector domain, against
// the same scalar loop test_quantize_bench measures. Same data, same row walk,
// same checksum -- only the produce loop differs, so the checksum must match
// exactly and the time is the answer.
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Mem;
import rayzor.Usize;
import rayzor.Bytes;

class Main {
    static inline var K = 4096;
    static inline var NROW = 64;
    static inline var ROWS = 40000;

    static function main():Void {
        var xb = Bytes.alloc(K * 4 * NROW);
        var xBase:Usize = xb.address();
        var ob = Bytes.alloc(K);
        var oBase:Usize = ob.address();

        var seed = 12345;
        for (i in 0...(K * NROW)) {
            seed = (seed * 1103515245 + 12345) & 0x3FFFFFFF;
            var f = (seed % 8000) / 1000.0 - 4.0;
            Mem.storeF32(xBase + Usize.fromInt(i << 2), f);
        }

        var checksum = 0;
        var t0 = Sys.time();
        for (r in 0...ROWS) {
            var row = xBase + Usize.fromInt(((r % NROW) * K) << 2);
            Mem.storeF32(row, (r % 17) * 0.25 - 2.0);
            quantizeVector(row, oBase, K);
            checksum = (checksum + rowSum(oBase, K)) & 0x3FFFFFFF;
        }
        var dt = Sys.time() - t0;

        Sys.println("checksum " + checksum);
        Sys.println("first8 " + head8(oBase));
        Sys.println("ms " + Std.int(dt * 1000.0));
    }

    static function quantizeVector(rowF:Usize, out:Usize, k:Int):Void {
        var mAcc = SIMD4f.splat(0.0);
        var j = 0;
        while (j < k) {
            var v = SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt(j << 2)));
            mAcc = mAcc.max(v.abs());
            j += 4;
        }
        var maxAbs = mAcc.get(0);
        var l1 = mAcc.get(1); if (l1 > maxAbs) maxAbs = l1;
        var l2 = mAcc.get(2); if (l2 > maxAbs) maxAbs = l2;
        var l3 = mAcc.get(3); if (l3 > maxAbs) maxAbs = l3;
        var invD = maxAbs > 0.0 ? 127.0 / maxAbs : 0.0;

        // 16 lanes per iteration: four f32x4 -> four i32x4 -> one i8x16.
        // The pack saturates at +/-128, one wider than the scalar clamp, so
        // clamp to +/-127 in the float domain to keep the two identical.
        var scale = SIMD4f.splat(invD);
        var hi = SIMD4f.splat(127.0);
        var lo = SIMD4f.splat(-127.0);
        var flip = SIMD16i8.splat(0x80);
        var jj = 0;
        while (jj < k) {
            var a = SIMD4i32.fromFloat(SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt(jj << 2))).mul(scale).clamp(lo, hi).round());
            var b = SIMD4i32.fromFloat(SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt((jj + 4) << 2))).mul(scale).clamp(lo, hi).round());
            var c = SIMD4i32.fromFloat(SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt((jj + 8) << 2))).mul(scale).clamp(lo, hi).round());
            var d = SIMD4i32.fromFloat(SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt((jj + 12) << 2))).mul(scale).clamp(lo, hi).round());
            // xor 0x80 is the +128 shift into u8: exact for every value in -128..127.
            SIMD16i8.xor(SIMD16i8.packI32(a, b, c, d), flip).store(Ptr.fromRaw(out + Usize.fromInt(jj)));
            jj += 16;
        }
    }

    static function rowSum(p:Usize, k:Int):Int {
        var s = 0;
        for (i in 0...k) s += Mem.loadU8(p + Usize.fromInt(i));
        return s;
    }

    static function head8(p:Usize):String {
        var s = "";
        for (i in 0...8) s += Mem.loadU8(p + Usize.fromInt(i)) + " ";
        return s;
    }
}
