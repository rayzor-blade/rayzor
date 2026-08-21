// Baseline of a three-way experiment on the int8 activation quantise. This is
// the produce loop as ordinary scalar Haxe, where `Float` means f64:
//
//   test_quantize_bench   this file   scalar, f64 arithmetic
//   test_quantize_single              scalar, Single (f32) arithmetic
//   test_quantize_vector              SIMD convert + saturating pack
//
// All three print the same checksum over the quantised bytes, so a faster one
// has to be byte-for-byte identical to count.
import rayzor.SIMD4f;
import rayzor.Ptr;
import rayzor.Mem;
import rayzor.Usize;
import rayzor.Bytes;

class test_quantize_bench {
    static inline var K = 4096;
    static inline var NROW = 64;
    static inline var ROWS = 40000;

    static function main():Void {
        var xb = Bytes.alloc(K * 4 * NROW);
        var xBase:Usize = xb.address();
        var ob = Bytes.alloc(K);
        var oBase:Usize = ob.address();

        // Deterministic pseudo-random activations in roughly [-4, 4).
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
            // Perturb the row so no iteration can be hoisted as loop-invariant.
            Mem.storeF32(row, (r % 17) * 0.25 - 2.0);
            quantizeScalar(row, oBase, K);
            checksum = (checksum + rowSum(oBase, K)) & 0x3FFFFFFF;
        }
        var dt = Sys.time() - t0;

        Sys.println("checksum " + checksum);
        Sys.println("first8 " + head8(oBase));
        Sys.println("ms " + Std.int(dt * 1000.0));
    }

    /** absmax over the row, then the scalar produce loop under measurement. */
    static function quantizeScalar(rowF:Usize, out:Usize, k:Int):Void {
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

        for (jj in 0...k) {
            var v = Mem.loadF32(rowF + Usize.fromInt(jj << 2)) * invD;
            var q = v >= 0.0 ? Std.int(v + 0.5) : Std.int(v - 0.5);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            Mem.storeU8(out + Usize.fromInt(jj), (q + 128) & 0xFF);
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
