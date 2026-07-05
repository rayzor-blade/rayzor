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
import rayzor.Mem;
import rayzor.concurrent.SpinPool;
import rayzor.concurrent.CpuTopology;

/**
 * Q4_K_M × Q8_K quantized matmul (gated by RAYZOR_HAXE_MATMUL).
 *
 * Kernel: per 256-weight super-block, 8 SDOT sub-block dots into 4 i32x4
 * accumulators (`SIMD4i32.dot`); weight scales/mins decode inline from the
 * block header (no O(weight) pre-decode buffer). Output rows are distributed
 * across the `SpinPool`; each row is computed by exactly one claimant with a
 * fixed reduction order, so results are bit-identical to the serial loop.
 * Per-call scratch is one activation batch of Q8_K quants, freed on return.
 *
 * Layout (matches QTensor): weight = 144-byte Q4_K super-blocks (Q6_K: 210),
 * row-major [out=rows, in=cols], blocks-per-row = cols>>8. Activations are
 * quantised to Q8_K here.
 */
class Q4Matmul {
    static inline function f16ToF32(bits:Int):Float {
        var sign:Int = (bits >> 15) & 1; var exp:Int = (bits >> 10) & 0x1F; var mant:Int = bits & 0x3FF;
        var sgn = (sign == 1) ? -1.0 : 1.0;
        if (exp == 0) return sgn * mant * 0.000000059604644775390625;
        if (exp == 31) return (mant == 0) ? sgn * 1e38 : 0.0;
        // Exact IEEE f16→f32 rebase: same sign/mantissa, exponent re-biased
        // 15→127. Branch-free bit construction (was an iterative pow2 loop).
        return Mem.f32FromBits((sign << 31) | ((exp + 112) << 23) | (mant << 13));
    }

    /** Quantise one activation row x[xBase .. xBase+K] to Q8_K (qs/bsums/dOut).
        `bsums` holds K/16 int32 group sums. */
    static inline function loadI32(bytes:Bytes, byteOff:Int):Int {
        return bytes.loadI32AlignedUnchecked(byteOff);
    }

    static inline function storeI32(bytes:Bytes, byteOff:Int, value:Int):Void {
        bytes.storeI32AlignedUnchecked(byteOff, value);
    }

    /** Quantize super-block `g` of a packed activation batch to Q8_K.
        Layout is g-linear: elements at g*256, bsums at g*64 bytes, scale at
        dOut[g]. `xBase`/`qsBase` are raw addresses of contiguous F32 input /
        Q8 scratch; all element accesses are inline typed loads/stores.
        Blocks are independent — safe to run concurrently across a band. */
    static inline function quantizeBlock(xBase:Usize, qsBase:Usize, bsums:Bytes,
            dOut:Array<Float>, g:Int):Void {
        var off = g * 256;
        var maxAbs = 0.0;
        for (i in 0...256) {
            var a = Mem.loadF32(xBase + Usize.fromInt((off + i) << 2));
            if (a < 0) a = -a; if (a > maxAbs) maxAbs = a;
        }
        if (maxAbs == 0.0) {
            dOut[g] = 0.0;
            for (i in 0...256) Mem.storeU8(qsBase + Usize.fromInt(off + i), 0);
            for (s in 0...16) storeI32(bsums, g * 64 + s * 4, 0);
        } else {
            var d = maxAbs / 127.0; dOut[g] = d; var invD = 127.0 / maxAbs;
            for (s in 0...16) {
                var sum = 0;
                for (j in 0...16) {
                    var v = Mem.loadF32(xBase + Usize.fromInt((off + s * 16 + j) << 2)) * invD;
                    var qf = (v >= 0) ? Math.floor(v + 0.5) : Math.ceil(v - 0.5);
                    var q = (qf > 127) ? 127 : ((qf < -128) ? -128 : qf);
                    Mem.storeU8(qsBase + Usize.fromInt(off + s * 16 + j), q & 0xFF); sum += q;
                }
                storeI32(bsums, g * 64 + (s << 2), sum);
            }
        }
    }

    /** Quantize `total` super-blocks, banded across the pool when it pays
        (blocks are independent; per-block work ~1-3us). */
    static function quantizeAll(xBase:Usize, qsBase:Usize, bsums:Bytes,
            dOut:Array<Float>, total:Int, sp:Null<SpinPool>):Void {
        if (sp != null && total >= 16) {
            var qband = function(lo:Int, hi:Int, node:Int):Void {
                for (g in lo...hi) quantizeBlock(xBase, qsBase, bsums, dOut, g);
            };
            sp.parallelRows(total, qband);
        } else {
            for (g in 0...total) quantizeBlock(xBase, qsBase, bsums, dOut, g);
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
        var imin = mn0 * (loadI32(bsums, b4) + loadI32(bsums, b4 + 4))
                 + mn1 * (loadI32(bsums, b4 + 8) + loadI32(bsums, b4 + 12))
                 + mn2 * (loadI32(bsums, b4 + 16) + loadI32(bsums, b4 + 20))
                 + mn3 * (loadI32(bsums, b4 + 24) + loadI32(bsums, b4 + 28))
                 + mn4 * (loadI32(bsums, b4 + 32) + loadI32(bsums, b4 + 36))
                 + mn5 * (loadI32(bsums, b4 + 40) + loadI32(bsums, b4 + 44))
                 + mn6 * (loadI32(bsums, b4 + 48) + loadI32(bsums, b4 + 52))
                 + mn7 * (loadI32(bsums, b4 + 56) + loadI32(bsums, b4 + 60));
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
                sumTerm2 += 32.0 * dScLo * loadI32(bsums, bi * 4)
                          + 32.0 * dScHi * loadI32(bsums, (bi + 1) * 4);
            }
        }
        return xd * (sumTerm1 - sumTerm2);
    }

    // One worker per logical CPU (CpuTopology reports a single NUMA node on
    // UMA hardware, which would run the bands serially); overridable with
    // RAYZOR_HAXE_MATMUL_WORKERS.
    public static function workerCount():Int {
        var n = CpuTopology.cpuCount();
        var env = Sys.getEnv("RAYZOR_HAXE_MATMUL_WORKERS");
        if (env != null) {
            var v = Std.parseInt(env);
            if (v != null && v > 0) n = v;
        }
        if (n < 1) n = 1;
        return n;
    }



    /**
     * y = x @ qw.T, x:[batch,K] F32, qw:[rows,K] Q4_K_M → y:[batch,rows] F32.
     * Output rows are banded once across the pool; batch is the INNER loop so
     * each quantized weight block is read from memory exactly once per output
     * row regardless of batch — the kernel is weight-bandwidth-bound, so a
     * batch-outer nesting made prefill cost seq × decode. All activation rows
     * are Q8_K-quantized up front into packed scratch (batch × one row).
     */
    public static function matmul(qw:QTensor, x:Tensor, ?sp:SpinPool):Tensor {
        var K = qw.cols();
        var bpr = K >> 8;
        var batch = Std.int(x.numel() / K);
        if (batch < 1) batch = 1;

        var qs = Bytes.alloc(batch * K);
        var bsums = Bytes.alloc(batch * bpr * 16 * 4);
        var dScales = new Array<Float>();
        dScales.resize(batch * bpr);
        var aBase = qs.address();
        // Contiguous F32 activation base — quantize reads it via inline
        // Mem.loadF32 instead of a per-element getFlat extern.
        var xBase = x.data().raw();
        var _tq = Sys.time();
        quantizeAll(xBase, aBase, bsums, dScales, batch * bpr, sp);
        if (sp != null) sp.addQuantUs(Std.int((Sys.time() - _tq) * 1e6));

        var y = runBanded(qw, batch, K, aBase, bsums, dScales, sp);

        qs.free();
        bsums.free();
        return y;
    }

    /** One banded pass of `qw` against pre-quantized Q8_K activations
        (packed batch rows at `aBase`/`bsums`/`dScales`). Shared by `matmul`
        and the fused batch path so a common activation is quantized once. */
    static function runBanded(qw:QTensor, batch:Int, K:Int, aBase:Usize, bsums:Bytes,
            dScales:Array<Float>, sp:Null<SpinPool>):Tensor {
        var bpr = K >> 8;
        var rows = qw.rows();
        var wBase = qw.dataPtr();
        // Q4_K_M promotes accuracy-sensitive tensors (attn_v, ffn_down) to
        // Q6_K (210-byte super-block). Dispatch to the matching per-block
        // dot; the Q8_K activation side is scheme-agnostic.
        var isQ6 = (qw.scheme() == QScheme.Q6_K);
        var blockBytes = isQ6 ? 210 : 144;

        var y = Tensor.zeros([batch, rows], DType.F32);
        // Inline f32 stores via Mem.storeF32 (narrows f64→f32 at the
        // boundary). NOT `y.data():Ptr<Float>` + write(): Float is f64, so
        // that store lands 8 bytes at an 8-byte stride and corrupts F32.
        var yBase = y.data().raw();

        var band = function(n0:Int, n1:Int, node:Int):Void {
            if (batch == 1) {
                // Decode fast path: scalar accumulator, no batch indexing.
                for (n in n0...n1) {
                    var sum = 0.0;
                    for (b in 0...bpr) {
                        var blk = n * bpr + b;
                        var xdb = dScales[b];
                        sum += isQ6
                            ? q6DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb)
                            : q4DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb);
                    }
                    Mem.storeF32(yBase + Usize.fromInt(n << 2), sum);
                }
                return;
            }
            // Batch tiled by 4 with SCALAR accumulators: an Array<Float>
            // accumulator writes through a per-element extern (bpr × batch
            // per row — ~1M calls per prefill matmul); scalars live in
            // registers. A row's weight blocks (bpr × blockBytes ≤ ~5KB)
            // stay L1-resident across the ≤ ceil(batch/4) re-walks, and
            // per-(row,batch) accumulation order over b is unchanged, so
            // outputs remain bit-identical.
            for (n in n0...n1) {
                var rt = 0;
                while (rt + 4 <= batch) {
                    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
                    for (b in 0...bpr) {
                        var blkOff = (n * bpr + b) * blockBytes;
                        var a0 = rt * K + b * 256;
                        var i0 = (rt * bpr + b) * 16;
                        if (isQ6) {
                            s0 += q6DotMA4(wBase, blkOff, aBase, a0, bsums, i0, dScales[rt * bpr + b]);
                            s1 += q6DotMA4(wBase, blkOff, aBase, a0 + K, bsums, i0 + bpr * 16, dScales[(rt + 1) * bpr + b]);
                            s2 += q6DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsums, i0 + 2 * bpr * 16, dScales[(rt + 2) * bpr + b]);
                            s3 += q6DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsums, i0 + 3 * bpr * 16, dScales[(rt + 3) * bpr + b]);
                        } else {
                            s0 += q4DotMA4(wBase, blkOff, aBase, a0, bsums, i0, dScales[rt * bpr + b]);
                            s1 += q4DotMA4(wBase, blkOff, aBase, a0 + K, bsums, i0 + bpr * 16, dScales[(rt + 1) * bpr + b]);
                            s2 += q4DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsums, i0 + 2 * bpr * 16, dScales[(rt + 2) * bpr + b]);
                            s3 += q4DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsums, i0 + 3 * bpr * 16, dScales[(rt + 3) * bpr + b]);
                        }
                    }
                    Mem.storeF32(yBase + Usize.fromInt((rt * rows + n) << 2), s0);
                    Mem.storeF32(yBase + Usize.fromInt(((rt + 1) * rows + n) << 2), s1);
                    Mem.storeF32(yBase + Usize.fromInt(((rt + 2) * rows + n) << 2), s2);
                    Mem.storeF32(yBase + Usize.fromInt(((rt + 3) * rows + n) << 2), s3);
                    rt += 4;
                }
                while (rt < batch) {
                    var sum = 0.0;
                    for (b in 0...bpr) {
                        var blkOff = (n * bpr + b) * blockBytes;
                        var xdb = dScales[rt * bpr + b];
                        sum += isQ6
                            ? q6DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsums, (rt * bpr + b) * 16, xdb)
                            : q4DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsums, (rt * bpr + b) * 16, xdb);
                    }
                    Mem.storeF32(yBase + Usize.fromInt((rt * rows + n) << 2), sum);
                    rt++;
                }
            }
        };
        // Persistent pool when the caller provides one (a spawn-per-call
        // pool pays thread create/join per matmul and nets zero); inline
        // otherwise. The pool is plumbed as an instance — a cross-module
        // OBJECT static reads garbage (statics aren't forwarded across
        // modules; see bugs_import_xmodule_member_resolution).
        if (sp != null) sp.parallelRows(rows, band);
        else band(0, rows, 0);
        return y;
    }

    /**
     * Fused multi-weight matmul over a SHARED activation: quantize x to Q8_K
     * exactly once, then one banded pass over the CONCATENATED row space of
     * up to three same-K weights (e.g. gate+up). N quantize passes and N
     * pool dispatches become 1+1; per-row math and reduction order are
     * identical to `matmul`, so outputs are bit-identical to unfused calls.
     * batch>1 inputs fall back to per-weight `matmul`. Returns one
     * [1, rows_i] F32 tensor per weight.
     */
    public static function matmulFused(w0:QTensor, w1:QTensor, w2:QTensor, x:Tensor, ?sp:SpinPool):Array<Tensor> {
        var K = w0.cols();
        var batch = Std.int(x.numel() / K);
        if (batch != 1) {
            // Prefill: quantize the shared activation ONCE, then one banded
            // pass per weight against the packed scratch.
            var bprB = K >> 8;
            var qsB = Bytes.alloc(batch * K);
            var bsumsB = Bytes.alloc(batch * bprB * 16 * 4);
            var dScalesB = new Array<Float>();
            dScalesB.resize(batch * bprB);
            var aB = qsB.address();
            var xB = x.data().raw();
            var _tqB = Sys.time();
            quantizeAll(xB, aB, bsumsB, dScalesB, batch * bprB, sp);
            if (sp != null) sp.addQuantUs(Std.int((Sys.time() - _tqB) * 1e6));
            var outs = [
                runBanded(w0, batch, K, aB, bsumsB, dScalesB, sp),
                runBanded(w1, batch, K, aB, bsumsB, dScalesB, sp)
            ];
            if (w2 != null) outs.push(runBanded(w2, batch, K, aB, bsumsB, dScalesB, sp));
            qsB.free();
            bsumsB.free();
            return outs;
        }
        var bpr = K >> 8;

        var qs = Bytes.alloc(K);
        var bsums = Bytes.alloc(bpr * 16 * 4);
        var dScales = new Array<Float>();
        dScales.resize(bpr);
        var aBase = qs.address();
        var xBase = x.data().raw();
        var _tq = Sys.time();
        quantizeAll(xBase, aBase, bsums, dScales, bpr, sp);
        if (sp != null) sp.addQuantUs(Std.int((Sys.time() - _tq) * 1e6));

        var r0 = w0.rows();
        var r1 = w1.rows();
        var r2 = (w2 != null) ? w2.rows() : 0;
        var e0 = r0;          // exclusive end of w0's global rows
        var e1 = r0 + r1;     // exclusive end of w1's
        var total = r0 + r1 + r2;

        var wb0 = w0.dataPtr();
        var wb1 = w1.dataPtr();
        var wb2 = (w2 != null) ? w2.dataPtr() : wb0;
        var q60 = (w0.scheme() == QScheme.Q6_K);
        var q61 = (w1.scheme() == QScheme.Q6_K);
        var q62 = (w2 != null) && (w2.scheme() == QScheme.Q6_K);

        var y0 = Tensor.zeros([1, r0], DType.F32);
        var y1 = Tensor.zeros([1, r1], DType.F32);
        var y2 = (w2 != null) ? Tensor.zeros([1, r2], DType.F32) : null;
        var yb0 = y0.data().raw();
        var yb1 = y1.data().raw();
        var yb2 = (w2 != null) ? y2.data().raw() : yb0;

        var band = function(n0:Int, n1:Int, node:Int):Void {
            for (g in n0...n1) {
                // Global row -> (weight, local row, base, scheme, out).
                var wBase:Usize; var isQ6:Bool; var n:Int; var yb:Usize;
                if (g < e0)      { wBase = wb0; isQ6 = q60; n = g;      yb = yb0; }
                else if (g < e1) { wBase = wb1; isQ6 = q61; n = g - e0; yb = yb1; }
                else             { wBase = wb2; isQ6 = q62; n = g - e1; yb = yb2; }
                var blockBytes = isQ6 ? 210 : 144;
                var sum = 0.0;
                for (b in 0...bpr) {
                    var blk = n * bpr + b;
                    var xdb = dScales[b];
                    sum += isQ6
                        ? q6DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb)
                        : q4DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb);
                }
                Mem.storeF32(yb + Usize.fromInt(n << 2), sum);
            }
        };
        if (sp != null) sp.parallelRows(total, band);
        else band(0, total, 0);

        qs.free();
        bsums.free();
        var outs = [y0, y1];
        if (w2 != null) outs.push(y2);
        return outs;
    }
}
