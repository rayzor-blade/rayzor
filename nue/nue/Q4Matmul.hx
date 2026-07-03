package nue;

import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
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
    static function q4DotMA4(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsums:Bytes, bsBase:Int, xd:Float):Float {
        var hdr = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wBlk)));
        var h0 = hdr.get(0) & 0xFF; var h1 = hdr.get(1) & 0xFF; var h2 = hdr.get(2) & 0xFF; var h3 = hdr.get(3) & 0xFF;
        var h4 = hdr.get(4) & 0xFF; var h5 = hdr.get(5) & 0xFF; var h6 = hdr.get(6) & 0xFF; var h7 = hdr.get(7) & 0xFF;
        var h8 = hdr.get(8) & 0xFF; var h9 = hdr.get(9) & 0xFF; var h10 = hdr.get(10) & 0xFF; var h11 = hdr.get(11) & 0xFF;
        var h12 = hdr.get(12) & 0xFF; var h13 = hdr.get(13) & 0xFF; var h14 = hdr.get(14) & 0xFF; var h15 = hdr.get(15) & 0xFF;
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

    // Forced-node pool, built once. `WorkerPool.global()` derives its worker
    // count from CPU topology, which reports a single node on UMA hardware
    // (Apple Silicon) and would run the row bands serially; force one worker
    // per logical CPU so the row loop actually fans out. Overridable with
    // RAYZOR_HAXE_MATMUL_WORKERS.
    static var _pool:WorkerPool = null;

    static function workers():WorkerPool {
        if (_pool == null) {
            var n = CpuTopology.cpuCount();
            var env = Sys.getEnv("RAYZOR_HAXE_MATMUL_WORKERS");
            if (env != null) {
                var v = Std.parseInt(env);
                if (v != null && v > 0) n = v;
            }
            if (n < 1) n = 1;
            _pool = WorkerPool.withForcedNodes(n);
        }
        return _pool;
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

        var qs = Bytes.alloc(K);
        var bsums = Bytes.alloc(bpr * 16 * 4);
        var dBytes = Bytes.alloc(bpr * 8);

        var y = Tensor.zeros([batch, rows], DType.F32);
        var yPtr = y.data();
        var pool = workers();

        for (r in 0...batch) {
            quantizeQ8K(x, r * K, K, qs, bsums, dBytes);
            var aBase = qs.address();
            var ob = r * rows;
            pool.parallelRows(rows, function(n0:Int, n1:Int, node:Int):Void {
                for (n in n0...n1) {
                    var sum = 0.0;
                    for (b in 0...bpr) {
                        var blk = n * bpr + b;
                        sum += q4DotMA4(wBase, blk * 144, aBase, b * 256, bsums, b * 16, dBytes.getDouble(b * 8));
                    }
                    var slot:Ptr<Float> = yPtr.offset(ob + n);
                    slot.write(sum);
                }
            });
        }

        qs.free();
        bsums.free();
        dBytes.free();
        return y;
    }
}
