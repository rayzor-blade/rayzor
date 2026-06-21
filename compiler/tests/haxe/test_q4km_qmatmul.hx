// Phase 4 gate: a full pure-Haxe Q4_K_M quantized matmul, validated against the
// Rust reference (QTensor.matmulXTQ) on a REAL Q4_K_M QTensor — the dot leaves
// per-block Rust FFI. Composes everything landed this session:
//   - QTensor.dataPtr()        — read weight super-blocks in-guest
//   - SIMD16i8.load/get        — block header (d/dmin/scales) + nibble bytes
//   - SIMD16i8.and/ushr        — Q4 nibble unpack
//   - SIMD4i32.dot (-> SDOT)   — the fused widening integer dot
//   - Bytes.address()          — load the Q8_K activation in-guest
//   - in-Haxe f16 decode + Q8_K activation quant
//
// Runs bit-close (f32 precision; Haxe folds in f64 vs the oracle's f32) on BOTH
// native and wasm — the actual decode-latency target. A wasm run is the gate.
//
// Workarounds in use (real compiler bugs, see memory):
//   - Q8_K block scales kept in a Bytes (setDouble/getDouble): Array<Float>
//     element set/get truncates f64 on wasm.
//   - clamp via a ternary, not `if(c) v=k;`: the statement-form reassign inside
//     the nested quant loop fails wasm compilation.
//   - block header via SIMD16i8.get, not Ptr<Int>.deref (which segfaults).

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.ds.QTensor;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Bytes;

class Main {
    static function pow2(e:Int):Float {
        var r = 1.0;
        if (e >= 0) { for (i in 0...e) r *= 2.0; } else { for (i in 0...(-e)) r *= 0.5; }
        return r;
    }
    static function f16ToF32(bits:Int):Float {
        var sign:Int = (bits >> 15) & 1; var exp:Int = (bits >> 10) & 0x1F; var mant:Int = bits & 0x3FF;
        var sgn = (sign == 1) ? -1.0 : 1.0;
        if (exp == 0) return sgn * mant * 0.000000059604644775390625;
        if (exp == 31) return (mant == 0) ? sgn * 1e38 : 0.0;
        return sgn * (1.0 + mant / 1024.0) * pow2(exp - 15);
    }

    // Q8_K activation quant (mirrors runtime quantize_block_q8_K): absmax scale,
    // round-ties-away, per-16 bsums. qs i8 in a Bytes; d per block in dOut (Bytes).
    static function quantizeQ8K(x:Tensor, K:Int, qs:Bytes, bsums:Array<Int>, dOut:Bytes):Void {
        var nb:Int = K >> 8;
        for (b in 0...nb) {
            var maxAbs = 0.0;
            for (i in 0...256) { var a = x.getFlat(b * 256 + i); if (a < 0) a = -a; if (a > maxAbs) maxAbs = a; }
            if (maxAbs == 0.0) {
                dOut.setDouble(b * 8, 0.0);
                for (i in 0...256) qs.set(b * 256 + i, 0);
                for (s in 0...16) bsums[b * 16 + s] = 0;
            } else {
                var d = maxAbs / 127.0; dOut.setDouble(b * 8, d); var invD = 127.0 / maxAbs;
                for (s in 0...16) {
                    var sum = 0;
                    for (j in 0...16) {
                        var v = x.getFlat(b * 256 + s * 16 + j) * invD;
                        var qf = (v >= 0) ? Math.floor(v + 0.5) : Math.ceil(v - 0.5);
                        var q = (qf > 127) ? 127 : ((qf < -128) ? -128 : qf);
                        qs.set(b * 256 + s * 16 + j, q & 0xFF); sum += q;
                    }
                    bsums[b * 16 + s] = sum;
                }
            }
        }
    }

    // 32-element sub-block integer dot: activation i8 (full range) · weight nibble
    // (0..15). The nibble is the dot's i7 operand (b) — activations span i8.
    static function subDot(wBase:Usize, wOff:Int, isHi:Bool, aBase:Usize, aOff:Int):Int {
        var acc = SIMD4i32.splat(0);
        for (half in 0...2) {
            var wRaw = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wOff + half * 16)));
            var wNib = isHi ? SIMD16i8.ushr(wRaw, 4) : SIMD16i8.and(wRaw, SIMD16i8.splat(0x0F));
            var aVec = SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(aOff + half * 16)));
            acc = SIMD4i32.dot(acc, aVec, wNib);
        }
        return acc.sum();
    }

    static function q4DotPtr(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsums:Array<Int>, bsBase:Int, xd:Float):Float {
        var hdr = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wBlk))); // d, dmin, scales[12]
        var d = f16ToF32((hdr.get(0) & 0xFF) | ((hdr.get(1) & 0xFF) << 8));
        var dmin = f16ToF32((hdr.get(2) & 0xFF) | ((hdr.get(3) & 0xFF) << 8));
        var acc = 0.0;
        for (s in 0...8) {
            var sc6:Int; var mn6:Int;
            if (s < 4) { sc6 = (hdr.get(4 + s) & 0xFF) & 63; mn6 = (hdr.get(4 + s + 4) & 0xFF) & 63; }
            else {
                var a:Int = hdr.get(4 + s + 4) & 0xFF; var bb:Int = hdr.get(4 + s - 4) & 0xFF; var c:Int = hdr.get(4 + s) & 0xFF;
                sc6 = (a & 0x0F) | (((bb >> 6) & 3) << 4); mn6 = ((a >> 4) & 0x0F) | (((c >> 6) & 3) << 4);
            }
            var subScale = d * sc6; var subMin = dmin * mn6;
            var p:Int = s >> 1; var isHi = (s & 1) == 1;
            var sdot = subDot(wBase, wBlk + 16 + p * 32, isHi, aBase, aBlk + s * 32);
            var bsum32 = bsums[bsBase + 2 * s] + bsums[bsBase + 2 * s + 1];
            acc += subScale * sdot - subMin * bsum32;
        }
        return xd * acc;
    }

    static function main() {
        var rows = 8; var K = 512; var bpr:Int = K >> 8; var nblk = rows * bpr;
        // Build a real Q4_K_M weight (varied nibbles per block) and wrap it.
        var wb = Bytes.alloc(nblk * 144);
        var scales = [1, 2, 3, 4, 1, 1, 1, 1, 21, 22, 23, 24];
        for (blk in 0...nblk) {
            var o = blk * 144;
            wb.set(o, 0x00); wb.set(o + 1, 0x3C); wb.set(o + 2, 0x00); wb.set(o + 3, 0x38);
            for (i in 0...12) wb.set(o + 4 + i, scales[i]);
            for (i in 0...128) wb.set(o + 16 + i, (i * (blk + 1)) & 0xFF);
        }
        var qt = QTensor.fromBytesQ4KM(wb, rows, K);
        var x = Tensor.rand([1, K], DType.F32);
        var yR = qt.matmulXTQ(x); // Rust reference

        var qs = Bytes.alloc(K); var bsums = []; for (i in 0...bpr * 16) bsums.push(0);
        var dBytes = Bytes.alloc(bpr * 8);
        quantizeQ8K(x, K, qs, bsums, dBytes);
        var wBase = qt.dataPtr(); var aBase = qs.address();

        var maxAbs = 0.0; var maxRel = 0.0;
        for (n in 0...rows) {
            var sum = 0.0;
            for (b in 0...bpr) sum += q4DotPtr(wBase, (n * bpr + b) * 144, aBase, b * 256, bsums, b * 16, dBytes.getDouble(b * 8));
            var rv = yR.getFlat(n);
            var e = sum - rv; if (e < 0) e = -e; if (e > maxAbs) maxAbs = e;
            var den = (rv < 0) ? -rv : rv; if (den > 0.5) { var r = e / den; if (r > maxRel) maxRel = r; }
        }
        // f64-vs-f32 fold tolerance; the integer dot is exact.
        if (maxRel < 0.0001 && maxAbs < 0.1) {
            Sys.println("PASS q4km-qmatmul rows=" + rows + " K=" + K + " maxRelErr=" + maxRel + " maxAbsErr=" + maxAbs);
        } else {
            Sys.println("FAIL q4km-qmatmul maxRelErr=" + maxRel + " maxAbsErr=" + maxAbs);
        }
    }
}
