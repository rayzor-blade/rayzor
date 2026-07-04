package nue;

import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
import rayzor.ds.QScheme;
import rayzor.ds.DType;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Bytes;
import rayzor.concurrent.WorkerPool;
import rayzor.concurrent.CpuTopology;

/**
 * Pure-Haxe Q4_K_M × Q8_K matmul — the native-parity alternative to the Rust
 * FFI kernel (`QTensor.matmulXTQThreaded`). Bit-exact vs FFI on synthetic blocks
 * (test_q4km_qmatmul, maxRelErr=0.0); wired onto a real `QTensor` weight so
 * `Linear.forward` can A/B it (gated by RAYZOR_HAXE_MATMUL).
 *
 * Kernel = multi-accumulator (4 i32x4 accs over 8 sub-blocks) + SDOT
 * (`SIMD4i32.dot` → VectorDot → AArch64 SDOT on the LLVM tier). Native only:
 * use `--llvm --release` (the LLVM tier emits SDOT). On wasm the Rust FFI
 * kernel is faster — keep it.
 *
 * Weight scales/mins are decoded INLINE per super-block (no O(weight)
 * pre-decode buffer), and output rows are banded across `WorkerPool` — one
 * output row is computed by exactly one worker, so results are bit-identical
 * to the serial reduction. Only fixed per-call scratch (one activation row's
 * Q8_K quants) is allocated, and it is freed before return.
 *
 * Layout (matches QTensor exactly): weight = 144-byte Q4_K_M super-blocks,
 * row-major [out=rows, in=cols], blocks-per-row = cols>>8. Activation is
 * quantised to Q8_K here (matching the FFI path's internal quant).
 */
class Q4Matmul {
    static inline function pow2(e:Int):Float {
        var r = 1.0;
        if (e >= 0) { for (i in 0...e) r *= 2.0; } else { for (i in 0...(-e)) r *= 0.5; }
        return r;
    }

    static inline function f16ToF32(bits:Int):Float {
        var sign:Int = (bits >> 15) & 1; var exp:Int = (bits >> 10) & 0x1F; var mant:Int = bits & 0x3FF;
        var sgn = (sign == 1) ? -1.0 : 1.0;
        if (exp == 0) return sgn * mant * 0.000000059604644775390625;
        if (exp == 31) return (mant == 0) ? sgn * 1e38 : 0.0;
        return sgn * (1.0 + mant / 1024.0) * pow2(exp - 15);
    }

    /** Quantise one activation row x[xBase .. xBase+K] to Q8_K (qs/bsums/dOut).
        `bsums` holds K/16 int32 group sums. */
    static function quantizeQ8K(x:Tensor, xBase:Int, K:Int, qs:Bytes, bsums:Bytes, dOut:Bytes):Void {
        var nb:Int = K >> 8;
        for (b in 0...nb) {
            var maxAbs = 0.0;
            for (i in 0...256) { var a = x.getFlat(xBase + b * 256 + i); if (a < 0) a = -a; if (a > maxAbs) maxAbs = a; }
            if (maxAbs == 0.0) {
                dOut.setDouble(b * 8, 0.0);
                for (i in 0...256) qs.set(b * 256 + i, 0);
                for (s in 0...16) bsums.setInt32((b * 16 + s) * 4, 0);
            } else {
                var d = maxAbs / 127.0; dOut.setDouble(b * 8, d); var invD = 127.0 / maxAbs;
                for (s in 0...16) {
                    var sum = 0;
                    for (j in 0...16) {
                        var v = x.getFlat(xBase + b * 256 + s * 16 + j) * invD;
                        var qf = (v >= 0) ? Math.floor(v + 0.5) : Math.ceil(v - 0.5);
                        var q = (qf > 127) ? 127 : ((qf < -128) ? -128 : qf);
                        qs.set(b * 256 + s * 16 + j, q & 0xFF); sum += q;
                    }
                    bsums.setInt32((b * 16 + s) * 4, sum);
                }
            }
        }
    }

    /** 4-lane partial dot of one 32-quant sub-block (low or high nibble). */
    static inline function subDotVec(wBase:Usize, wOff:Int, isHi:Bool, aBase:Usize, aOff:Int):SIMD4i32 {
        var acc = SIMD4i32.splat(0);
        for (half in 0...2) {
            var wRaw = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wOff + half * 16)));
            var wNib = isHi ? SIMD16i8.ushr(wRaw, 4) : SIMD16i8.and(wRaw, SIMD16i8.splat(0x0F));
            var aVec = SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(aOff + half * 16)));
            acc = SIMD4i32.dot(acc, aVec, wNib);
        }
        return acc;
    }

    /** One super-block's f32 contribution: 8 sub-block dots, 4 accumulators.
        Decodes this block's f32 d/dmin and its 8 6-bit scale/min pairs inline
        from the 16-byte header — no shared pre-decode buffer, so it is safe to
        call concurrently across output-row bands. */
    static inline function q4DotMA4(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsums:Bytes, bsBase:Int, xd:Float):Float {
        var hdr = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wBlk)));
        var h0 = hdr[0] & 0xFF; var h1 = hdr[1] & 0xFF; var h2 = hdr[2] & 0xFF; var h3 = hdr[3] & 0xFF;
        var h4 = hdr[4] & 0xFF; var h5 = hdr[5] & 0xFF; var h6 = hdr[6] & 0xFF; var h7 = hdr[7] & 0xFF;
        var h8 = hdr[8] & 0xFF; var h9 = hdr[9] & 0xFF; var h10 = hdr[10] & 0xFF; var h11 = hdr[11] & 0xFF;
        var h12 = hdr[12] & 0xFF; var h13 = hdr[13] & 0xFF; var h14 = hdr[14] & 0xFF; var h15 = hdr[15] & 0xFF;
        var d = f16ToF32(h0 | (h1 << 8));
        var dmin = f16ToF32(h2 | (h3 << 8));
        var sc0 = h4 & 63; var sc1 = h5 & 63; var sc2 = h6 & 63; var sc3 = h7 & 63;
        var mn0 = h8 & 63; var mn1 = h9 & 63; var mn2 = h10 & 63; var mn3 = h11 & 63;
        var sc4 = (h12 & 0x0F) | (((h4 >> 6) & 3) << 4); var mn4 = ((h12 >> 4) & 0x0F) | (((h8 >> 6) & 3) << 4);
        var sc5 = (h13 & 0x0F) | (((h5 >> 6) & 3) << 4); var mn5 = ((h13 >> 4) & 0x0F) | (((h9 >> 6) & 3) << 4);
        var sc6 = (h14 & 0x0F) | (((h6 >> 6) & 3) << 4); var mn6 = ((h14 >> 4) & 0x0F) | (((h10 >> 6) & 3) << 4);
        var sc7 = (h15 & 0x0F) | (((h7 >> 6) & 3) << 4); var mn7 = ((h15 >> 4) & 0x0F) | (((h11 >> 6) & 3) << 4);

        var dv0 = subDotVec(wBase, wBlk + 16 + 0 * 32, false, aBase, aBlk + 0 * 32);
        var dv1 = subDotVec(wBase, wBlk + 16 + 0 * 32, true,  aBase, aBlk + 1 * 32);
        var dv2 = subDotVec(wBase, wBlk + 16 + 1 * 32, false, aBase, aBlk + 2 * 32);
        var dv3 = subDotVec(wBase, wBlk + 16 + 1 * 32, true,  aBase, aBlk + 3 * 32);
        var dv4 = subDotVec(wBase, wBlk + 16 + 2 * 32, false, aBase, aBlk + 4 * 32);
        var dv5 = subDotVec(wBase, wBlk + 16 + 2 * 32, true,  aBase, aBlk + 5 * 32);
        var dv6 = subDotVec(wBase, wBlk + 16 + 3 * 32, false, aBase, aBlk + 6 * 32);
        var dv7 = subDotVec(wBase, wBlk + 16 + 3 * 32, true,  aBase, aBlk + 7 * 32);
        var i0 = SIMD4i32.splat(sc0) * dv0 + SIMD4i32.splat(sc4) * dv4;
        var i1 = SIMD4i32.splat(sc1) * dv1 + SIMD4i32.splat(sc5) * dv5;
        var i2 = SIMD4i32.splat(sc2) * dv2 + SIMD4i32.splat(sc6) * dv6;
        var i3 = SIMD4i32.splat(sc3) * dv3 + SIMD4i32.splat(sc7) * dv7;
        var isum = (i0 + i1) + (i2 + i3);
        var b4 = bsBase * 4;
        var imin = mn0 * (bsums.getInt32(b4) + bsums.getInt32(b4 + 4))
                 + mn1 * (bsums.getInt32(b4 + 8) + bsums.getInt32(b4 + 12))
                 + mn2 * (bsums.getInt32(b4 + 16) + bsums.getInt32(b4 + 20))
                 + mn3 * (bsums.getInt32(b4 + 24) + bsums.getInt32(b4 + 28))
                 + mn4 * (bsums.getInt32(b4 + 32) + bsums.getInt32(b4 + 36))
                 + mn5 * (bsums.getInt32(b4 + 40) + bsums.getInt32(b4 + 44))
                 + mn6 * (bsums.getInt32(b4 + 48) + bsums.getInt32(b4 + 52))
                 + mn7 * (bsums.getInt32(b4 + 56) + bsums.getInt32(b4 + 60));
        return xd * (d * isum.sum() - dmin * imin);
    }

    /** One Q6_K super-block's f32 contribution (256 weights, 210-byte block).
        Rebuilds each 6-bit weight from its ql low-nibble + qh 2-bit pair and
        dots it against the shared Q8_K activation with SDOT; the unsigned
        0..63 → −32..31 bias folds into 32·Σx via the per-16 bsums. Mirrors
        runtime-core `vec_dot_q6_K_q8_K`. Safe to call concurrently across
        output-row bands (no shared scratch). */
    static inline function q6DotMA4(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsums:Bytes, bsBase:Int, xd:Float):Float {
        var scVec = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wBlk + 192)));
        var dVec = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wBlk + 194)));
        var d = f16ToF32((dVec[14] & 0xFF) | ((dVec[15] & 0xFF) << 8));
        var mask = SIMD16i8.splat(0x0F);
        var mask2 = SIMD16i8.splat(0x03);
        var sumTerm1 = 0.0;
        var sumTerm2 = 0.0;
        for (n in 0...2) {
            var qlB = wBlk + n * 64;
            var qhB = wBlk + 128 + n * 32;
            var scOff = n * 8;
            var outOff = n * 128;
            var ql0 = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(qlB)));
            var ql1 = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(qlB + 16)));
            var ql2 = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(qlB + 32)));
            var ql3 = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(qlB + 48)));
            var qh0 = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(qhB)));
            var qh1 = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(qhB + 16)));
            for (j in 0...4) {
                var qlP0 = (j == 0) ? SIMD16i8.and(ql0, mask)
                         : (j == 1) ? SIMD16i8.and(ql2, mask)
                         : (j == 2) ? SIMD16i8.ushr(ql0, 4) : SIMD16i8.ushr(ql2, 4);
                var qlP1 = (j == 0) ? SIMD16i8.and(ql1, mask)
                         : (j == 1) ? SIMD16i8.and(ql3, mask)
                         : (j == 2) ? SIMD16i8.ushr(ql1, 4) : SIMD16i8.ushr(ql3, 4);
                var qhP0 = (j == 0) ? SIMD16i8.and(qh0, mask2)
                         : (j == 1) ? SIMD16i8.and(SIMD16i8.ushr(qh0, 2), mask2)
                         : (j == 2) ? SIMD16i8.and(SIMD16i8.ushr(qh0, 4), mask2) : SIMD16i8.ushr(qh0, 6);
                var qhP1 = (j == 0) ? SIMD16i8.and(qh1, mask2)
                         : (j == 1) ? SIMD16i8.and(SIMD16i8.ushr(qh1, 2), mask2)
                         : (j == 2) ? SIMD16i8.and(SIMD16i8.ushr(qh1, 4), mask2) : SIMD16i8.ushr(qh1, 6);
                var qLo = SIMD16i8.or(qlP0, SIMD16i8.shl(qhP0, 4));
                var qHi = SIMD16i8.or(qlP1, SIMD16i8.shl(qhP1, 4));
                var xSpan = aBlk + outOff + j * 32;
                var xLo = SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(xSpan)));
                var xHi = SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(xSpan + 16)));
                var sdotLo = SIMD4i32.dot(SIMD4i32.splat(0), xLo, qLo).sum();
                var sdotHi = SIMD4i32.dot(SIMD4i32.splat(0), xHi, qHi).sum();
                var dScLo:Float = d * scVec.get(scOff + 2 * j);
                var dScHi:Float = d * scVec.get(scOff + 2 * j + 1);
                sumTerm1 += dScLo * sdotLo + dScHi * sdotHi;
                var bi = bsBase + ((outOff + j * 32) >> 4);
                sumTerm2 += 32.0 * dScLo * bsums.getInt32(bi * 4)
                          + 32.0 * dScHi * bsums.getInt32((bi + 1) * 4);
            }
        }
        return xd * (sumTerm1 - sumTerm2);
    }

    // `WorkerPool.global()` derives its worker count from CPU topology, which
    // reports a single node on UMA hardware (Apple Silicon) and would run the
    // row bands serially; force one worker per logical CPU so the row loop
    // actually fans out. Overridable with RAYZOR_HAXE_MATMUL_WORKERS. Built
    // per call (a WorkerPool is just a node-count holder, and parallelRows
    // spawns fresh threads each call regardless) — NOT via a static field,
    // whose mutations don't reliably survive across calls on the JIT baseline
    // tier and left the pool receiver reading as garbage.
    static function workers():WorkerPool {
        var n = CpuTopology.cpuCount();
        var env = Sys.getEnv("RAYZOR_HAXE_MATMUL_WORKERS");
        if (env != null) {
            var v = Std.parseInt(env);
            if (v != null && v > 0) n = v;
        }
        if (n < 1) n = 1;
        return WorkerPool.withForcedNodes(n);
    }

    /**
     * y = x @ qw.T, x:[batch,K] F32, qw:[rows,K] Q4_K_M → y:[batch,rows] F32.
     * Output rows are banded across `WorkerPool.global()`; per-call heap is one
     * activation row of Q8_K scratch plus the returned tensor (no O(weight)
     * pre-decode buffer).
     */
    public static function matmul(qw:QTensor, x:Tensor):Tensor {
        var rows = qw.rows();
        var K = qw.cols();
        var bpr = K >> 8;
        var wBase = qw.dataPtr();
        var batch = Std.int(x.numel() / K);
        if (batch < 1) batch = 1;
        // Q4_K_M GGUFs promote accuracy-sensitive tensors (attn_v, ffn_down)
        // to Q6_K, which has a different 210-byte super-block. Dispatch to the
        // matching per-block dot; the Q8_K activation side is scheme-agnostic.
        var isQ6 = (qw.scheme() == QScheme.Q6_K);
        var blockBytes = isQ6 ? 210 : 144;

        var qs = Bytes.alloc(K);
        var bsums = Bytes.alloc(bpr * 16 * 4);
        var dBytes = Bytes.alloc(bpr * 8);

        var y = Tensor.zeros([batch, rows], DType.F32);
        var pool = workers();

        for (r in 0...batch) {
            quantizeQ8K(x, r * K, K, qs, bsums, dBytes);
            var aBase = qs.address();
            var ob = r * rows;
            // Write results with the dtype-aware flat setter, NOT a raw
            // `y.data():Ptr<Float>` + `write()`: `Float` is f64, so the raw
            // store lands 8 bytes at an 8-byte stride and corrupts the F32
            // output buffer. setFlat narrows to the tensor's element type.
            pool.parallelRows(rows, function(n0:Int, n1:Int, node:Int):Void {
                for (n in n0...n1) {
                    var sum = 0.0;
                    for (b in 0...bpr) {
                        var blk = n * bpr + b;
                        var xdb = dBytes.getDouble(b * 8);
                        sum += isQ6
                            ? q6DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb)
                            : q4DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb);
                    }
                    y.setFlat(ob + n, sum);
                }
            });
        }

        qs.free();
        bsums.free();
        dBytes.free();
        return y;
    }
}
