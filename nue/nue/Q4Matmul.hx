package nue;

import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
import rayzor.ds.QScheme;
import rayzor.ds.DType;
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Bytes;
import rayzor.Mem;
import rayzor.concurrent.SpinPool;
import rayzor.concurrent.CpuTopology;

/**
 * Q4_K_M × Q8_K quantized matmul (gated by NUE_MATMUL).
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
    static var _useFused:Int = 0;
    static var _dumpFusedGate:Int = 0;

    public static function useFusedMatmul():Bool {
        if (_useFused == 0) {
            var v = Sys.getEnvOr("NUE_FUSED_MATMUL", "RAYZOR_HAXE_FUSED_MATMUL");
            // The fused gate/up/qkv path is still experimental. It reduces
            // dispatch count, but full llama-chat runs on macOS regressed
            // versus the split Haxe kernels, so keep it opt-in until the
            // fused row-space lowering is proven faster and bit-stable.
            _useFused = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
            if (_dumpFusedGate == 0) {
                var dump = Sys.getEnvOr("NUE_DUMP_Q4_GATES", "RAYZOR_DUMP_Q4_GATES");
                _dumpFusedGate = (dump != null && dump != "0" && dump != "" && dump != "false") ? 1 : 2;
            }
            if (_dumpFusedGate == 1) {
                Sys.println("[q4-gate] fused_matmul=" + (_useFused == 1 ? "on" : "off")
                    + " env=" + (v == null ? "<unset>" : v));
            }
        }
        return _useFused == 1;
    }

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
        Layout is g-linear: elements at g*256, bsums at g*64 bytes, scale as
        one f32 at dBase + g*4. `xBase`/`qsBase`/`dBase` are raw addresses of
        contiguous F32 input / Q8 scratch / f32 scale scratch; every element
        access is an inline typed load/store — the scale buffer is raw Bytes
        (not Array<Float>) so a hot row never touches the boxing element
        extern. Blocks are independent — safe to run concurrently across a band. */
    static inline function quantizeBlock(xBase:Usize, qsBase:Usize, bsums:Bytes,
            dBase:Usize, g:Int):Void {
        var off = g * 256;
        // SIMD maxAbs: 4-wide abs+max with 2 accumulators to break the
        // loop-carried max latency chain. IEEE max is exact + associative
        // and abs(v)==max(v,-v) for non-NaN activations, so the reduced
        // maxAbs is bit-identical to the scalar loop. The horizontal fold
        // is ternary-only (fcsel, no haxe_box_float).
        var pbase = xBase + Usize.fromInt(off << 2);
        var m0 = SIMD4f.splat(0.0);
        var m1 = SIMD4f.splat(0.0);
        var vi = 0;
        while (vi < 256) {
            var v0 = SIMD4f.load(Ptr.fromRaw(pbase + Usize.fromInt(vi << 2)));
            var v1 = SIMD4f.load(Ptr.fromRaw(pbase + Usize.fromInt((vi + 4) << 2)));
            m0 = m0.max(v0.abs());
            m1 = m1.max(v1.abs());
            vi += 8;
        }
        var m = m0.max(m1);
        var l0 = m.get(0); var l1 = m.get(1); var l2 = m.get(2); var l3 = m.get(3);
        var mx01 = l0 > l1 ? l0 : l1;
        var mx23 = l2 > l3 ? l2 : l3;
        var maxAbs = mx01 > mx23 ? mx01 : mx23;
        if (maxAbs == 0.0) {
            Mem.storeF32(dBase + Usize.fromInt(g << 2), 0.0);
            for (i in 0...256) Mem.storeU8(qsBase + Usize.fromInt(off + i), 0);
            for (s in 0...16) storeI32(bsums, g * 64 + s * 4, 0);
        } else {
            var d = maxAbs / 127.0;
            Mem.storeF32(dBase + Usize.fromInt(g << 2), d);
            var invD = 127.0 / maxAbs;
            for (s in 0...16) {
                var sum = 0;
                for (j in 0...16) {
                    var v = Mem.loadF32(xBase + Usize.fromInt((off + s * 16 + j) << 2)) * invD;
                    // Round-half-away via Std.int (fptosi intrinsic, no box).
                    // Math.floor/ceil boxed the Float argument per element —
                    // millions of heap allocs/token. Same result: truncation
                    // of v±0.5 equals floor(v+0.5)/ceil(v-0.5) by sign.
                    var q = v >= 0 ? Std.int(v + 0.5) : Std.int(v - 0.5);
                    if (q > 127) q = 127; else if (q < -128) q = -128;
                    Mem.storeU8(qsBase + Usize.fromInt(off + s * 16 + j), q & 0xFF); sum += q;
                }
                storeI32(bsums, g * 64 + (s << 2), sum);
            }
        }
    }

    /** Quantize `total` super-blocks, banded across the pool when it pays
        (blocks are independent; per-block work ~1-3us). */
    static function quantizeAll(xBase:Usize, qsBase:Usize, bsums:Bytes,
            dBase:Usize, total:Int, sp:Null<SpinPool>):Void {
        var minParallel = sp != null ? sp.workers() * 8 : 0;
        if (minParallel < 64) minParallel = 64;
        if (sp != null && total >= minParallel) {
            var qband = function(lo:Int, hi:Int, node:Int):Void {
                for (g in lo...hi) quantizeBlock(xBase, qsBase, bsums, dBase, g);
            };
            sp.parallelRows(total, qband);
        } else {
            for (g in 0...total) quantizeBlock(xBase, qsBase, bsums, dBase, g);
        }
    }

    /** 4-lane partial dot of one 32-quant sub-block (low or high nibble). */
    static inline function subDotVec(wBase:Usize, wOff:Int, isHi:Bool, aBase:Usize, aOff:Int):SIMD4i32 {
        var acc = SIMD4i32.splat(0);
        for (half in 0...2) {
            var wRaw = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wOff + half * 16)));
            var wNib = isHi ? SIMD16i8.ushr(wRaw, 4) : SIMD16i8.and(wRaw, SIMD16i8.splat(0x0F));
            var aVec = SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(aOff + half * 16)));
            acc = SIMD4i32.dotI8I7(acc, aVec, wNib);
        }
        return acc;
    }

    /** One super-block's f32 contribution: 8 sub-block dots, 4 accumulators.
        Decodes this block's f32 d/dmin and its 8 6-bit scale/min pairs inline
        from the 16-byte header — no shared pre-decode buffer, so it is safe to
        call concurrently across output-row bands. */
    static inline function q4DotMA4(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsums:Bytes, bsBase:Int, xd:Float):Float {
        // Header decode via four u32 loads + shifts (the llama.cpp kmask
        // shape) instead of 16 SIMD lane extracts — same bit math, fewer
        // and cheaper ops on the ~4M headers decoded per token.
        var w0 = Mem.loadI32(wBase + Usize.fromInt(wBlk));       // d | dmin<<16
        var u0 = Mem.loadI32(wBase + Usize.fromInt(wBlk + 4));   // sc0..3 bytes
        var u1 = Mem.loadI32(wBase + Usize.fromInt(wBlk + 8));   // mn0..3 bytes
        var u2 = Mem.loadI32(wBase + Usize.fromInt(wBlk + 12));  // packed lo4s
        var d = f16ToF32(w0 & 0xFFFF);
        var dmin = f16ToF32((w0 >>> 16) & 0xFFFF);
        var sc0 = u0 & 63; var sc1 = (u0 >>> 8) & 63; var sc2 = (u0 >>> 16) & 63; var sc3 = (u0 >>> 24) & 63;
        var mn0 = u1 & 63; var mn1 = (u1 >>> 8) & 63; var mn2 = (u1 >>> 16) & 63; var mn3 = (u1 >>> 24) & 63;
        var sc4 = (u2 & 0x0F) | (((u0 >>> 6) & 3) << 4);
        var mn4 = ((u2 >>> 4) & 0x0F) | (((u1 >>> 6) & 3) << 4);
        var sc5 = ((u2 >>> 8) & 0x0F) | (((u0 >>> 14) & 3) << 4);
        var mn5 = ((u2 >>> 12) & 0x0F) | (((u1 >>> 14) & 3) << 4);
        var sc6 = ((u2 >>> 16) & 0x0F) | (((u0 >>> 22) & 3) << 4);
        var mn6 = ((u2 >>> 20) & 0x0F) | (((u1 >>> 22) & 3) << 4);
        var sc7 = ((u2 >>> 24) & 0x0F) | (((u0 >>> 30) & 3) << 4);
        var mn7 = ((u2 >>> 28) & 0x0F) | (((u1 >>> 30) & 3) << 4);

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
                var sdotLo = SIMD4i32.dotI8I7(SIMD4i32.splat(0), xLo, qLo).sum();
                var sdotHi = SIMD4i32.dotI8I7(SIMD4i32.splat(0), xHi, qHi).sum();
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

    // Matmul pool policy — Nue owns it; the SpinPool primitive is env-agnostic
    // and receives the resolved config. Profiles trade peak for co-runner
    // robustness: throughput (all P-cores, default), cooperative (P-1, gentle
    // spin so idle workers cede cores), latency (P-1, minimal spin).
    public static function poolProfile():String {
        var p = Sys.getEnvOr("NUE_POOL_PROFILE", "RAYZOR_HAXE_POOL_PROFILE");
        if (p == null || p == "") return "throughput";
        return p;
    }

    // Claimants = physical P-cores (caller included), never past the P-cluster.
    // cooperative/latency leave one P-core free — that slack keeps a co-runner
    // (or the OS) from preempting a spinning worker / the coordinator, which
    // otherwise ~halves throughput under contention.
    public static function workerCount():Int {
        var n = CpuTopology.perfCoreCount();
        var prof = poolProfile();
        if ((prof == "cooperative" || prof == "latency") && n > 2) n = n - 1;
        var env = Sys.getEnvOr("NUE_MATMUL_WORKERS", "RAYZOR_HAXE_MATMUL_WORKERS");
        if (env != null) {
            var v:Int = Std.int(Std.parseFloat(env));
            if (v > 0) n = v;
        }
        if (n < 1) n = 1;
        return n;
    }

    /** Tight-spin budget; 0 = platform default. Lower = cede cores sooner. */
    public static function poolSpins():Int {
        var prof = poolProfile();
        var s:Int = 0;
        if (prof == "cooperative") s = 20000;
        else if (prof == "latency") s = 2000;
        var env = Sys.getEnvOr("NUE_POOL_SPINS", "RAYZOR_HAXE_POOL_SPINS");
        if (env != null) {
            var v:Int = Std.int(Std.parseFloat(env));
            if (v > 0) s = v;
        }
        return s;
    }

    /** Per-spin relax hint; -1 = platform default. */
    public static function poolRelax():Int {
        var env = Sys.getEnvOr("NUE_POOL_RELAX", "RAYZOR_HAXE_POOL_RELAX");
        if (env != null) return (env != "0" && env != "" && env != "false") ? 1 : 0;
        return -1;
    }

    /** Pool dispatch timing; 1 = on. */
    public static function poolProfiling():Int {
        var env = Sys.getEnvOr("NUE_PROFILE_POOL", "RAYZOR_PROFILE_POOL");
        return (env != null && env != "0" && env != "" && env != "false") ? 1 : 0;
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

        var pooled = sp != null;
        var qs = pooled ? sp.scratchBytes(0, batch * K) : Bytes.alloc(batch * K);
        var bsums = pooled ? sp.scratchBytes(1, batch * bpr * 16 * 4) : Bytes.alloc(batch * bpr * 16 * 4);
        var dScales = pooled ? sp.scratchBytes(2, batch * bpr * 4) : Bytes.alloc(batch * bpr * 4);
        var aBase = qs.address();
        var dBase = dScales.address();
        // Contiguous F32 activation base — quantize reads it via inline
        // Mem.loadF32 instead of a per-element getFlat extern.
        var xBase = x.data().raw();
        var _prof = sp != null && sp.profiling();
        var _tq = _prof ? Sys.time() : 0.0;
        quantizeAll(xBase, aBase, bsums, dBase, batch * bpr, sp);
        if (_prof) sp.addQuantUs(Std.int((Sys.time() - _tq) * 1e6));

        var y = runBanded(qw, batch, K, aBase, bsums, dBase, sp);

        if (!pooled) {
            qs.free();
            bsums.free();
            dScales.free();
        }
        return y;
    }

    /** One banded pass of `qw` against pre-quantized Q8_K activations
        (packed batch rows at `aBase`/`bsums`/`dBase`). Shared by `matmul`
        and the fused batch path so a common activation is quantized once.
        `dBase` is a raw f32 scale buffer (one f32 per super-block) — read
        via Mem.loadF32 so the hot row makes no boxing element-extern calls. */
    static function runBanded(qw:QTensor, batch:Int, K:Int, aBase:Usize, bsums:Bytes,
            dBase:Usize, sp:Null<SpinPool>):Tensor {
        var bpr = K >> 8;
        var rows = qw.rows();
        var wBase = qw.dataPtr();
        // Q4_K_M promotes accuracy-sensitive tensors (attn_v, ffn_down) to
        // Q6_K (210-byte super-block). Dispatch to the matching per-block
        // dot; the Q8_K activation side is scheme-agnostic.
        var isQ6 = (qw.scheme() == QScheme.Q6_K);
        var blockBytes = isQ6 ? 210 : 144;

        var y = Tensor.uninit([batch, rows], DType.F32);
        // Inline f32 stores via Mem.storeF32 (narrows f64→f32 at the
        // boundary). NOT `y.data():Ptr<Float>` + write(): Float is f64, so
        // that store lands 8 bytes at an 8-byte stride and corrupts F32.
        var yBase = y.data().raw();

        var band = function(n0:Int, n1:Int, node:Int):Void {
            if (batch == 1) {
                // Decode fast path: scalar accumulator, no batch indexing.
                // No software prefetch here: this model's active weight set
                // fits the last-level cache, so prefetches are pure overhead
                // (measured +8%/token). Revisit via Mem.prefetch when the
                // weight set exceeds LLC (single-thread streaming measured
                // 8.5 -> 13.5 GMAC/s with it).
                for (n in n0...n1) {
                    var sum = 0.0;
                    // Two-block pairing: adjacent blocks are independent, so
                    // evaluating b and b+1 into one add keeps ~16 SDOT chains
                    // in flight and hides each block's scalar tail (hsum +
                    // f64 scale math + mins fold) under the other's vector
                    // work — the shape of the Rust kernel's paired inner loop.
                    var base = n * bpr;
                    var b = 0;
                    if (isQ6) {
                        while (b + 2 <= bpr) {
                            sum += q6DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsums, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)))
                                 + q6DotMA4(wBase, (base + b + 1) * blockBytes, aBase, (b + 1) * 256, bsums, (b + 1) * 16, Mem.loadF32(dBase + Usize.fromInt((b + 1) << 2)));
                            b += 2;
                        }
                        while (b < bpr) {
                            sum += q6DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsums, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)));
                            b++;
                        }
                    } else {
                        while (b + 2 <= bpr) {
                            sum += q4DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsums, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)))
                                 + q4DotMA4(wBase, (base + b + 1) * blockBytes, aBase, (b + 1) * 256, bsums, (b + 1) * 16, Mem.loadF32(dBase + Usize.fromInt((b + 1) << 2)));
                            b += 2;
                        }
                        while (b < bpr) {
                            sum += q4DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsums, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)));
                            b++;
                        }
                    }
                    Mem.storeF32(yBase + Usize.fromInt(n << 2), sum);
                }
                return;
            }
            // Batch tiled by 4 with SCALAR accumulators (registers, not a
            // heap Array). A row's weight blocks (bpr × blockBytes ≤ ~5KB)
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
                        var d0 = Mem.loadF32(dBase + Usize.fromInt((rt * bpr + b) << 2));
                        var d1 = Mem.loadF32(dBase + Usize.fromInt(((rt + 1) * bpr + b) << 2));
                        var d2 = Mem.loadF32(dBase + Usize.fromInt(((rt + 2) * bpr + b) << 2));
                        var d3 = Mem.loadF32(dBase + Usize.fromInt(((rt + 3) * bpr + b) << 2));
                        if (isQ6) {
                            s0 += q6DotMA4(wBase, blkOff, aBase, a0, bsums, i0, d0);
                            s1 += q6DotMA4(wBase, blkOff, aBase, a0 + K, bsums, i0 + bpr * 16, d1);
                            s2 += q6DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsums, i0 + 2 * bpr * 16, d2);
                            s3 += q6DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsums, i0 + 3 * bpr * 16, d3);
                        } else {
                            s0 += q4DotMA4(wBase, blkOff, aBase, a0, bsums, i0, d0);
                            s1 += q4DotMA4(wBase, blkOff, aBase, a0 + K, bsums, i0 + bpr * 16, d1);
                            s2 += q4DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsums, i0 + 2 * bpr * 16, d2);
                            s3 += q4DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsums, i0 + 3 * bpr * 16, d3);
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
                        var xdb = Mem.loadF32(dBase + Usize.fromInt((rt * bpr + b) << 2));
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
        if (!useFusedMatmul()) {
            var split = [matmul(w0, x, sp), matmul(w1, x, sp)];
            if (w2 != null) split.push(matmul(w2, x, sp));
            return split;
        }
        var K = w0.cols();
        var batch = Std.int(x.numel() / K);
        if (batch != 1) {
            // Prefill: quantize the shared activation ONCE, then one banded
            // pass per weight against the packed scratch.
            var bprB = K >> 8;
            var pooledB = sp != null;
            var qsB = pooledB ? sp.scratchBytes(0, batch * K) : Bytes.alloc(batch * K);
            var bsumsB = pooledB ? sp.scratchBytes(1, batch * bprB * 16 * 4) : Bytes.alloc(batch * bprB * 16 * 4);
            var dScalesB = pooledB ? sp.scratchBytes(2, batch * bprB * 4) : Bytes.alloc(batch * bprB * 4);
            var aB = qsB.address();
            var dB = dScalesB.address();
            var xB = x.data().raw();
            var _profB = sp != null && sp.profiling();
            var _tqB = _profB ? Sys.time() : 0.0;
            quantizeAll(xB, aB, bsumsB, dB, batch * bprB, sp);
            if (_profB) sp.addQuantUs(Std.int((Sys.time() - _tqB) * 1e6));
            var outs = runBandedFused(w0, w1, w2, batch, K, aB, bsumsB, dB, sp);
            if (!pooledB) {
                qsB.free();
                bsumsB.free();
                dScalesB.free();
            }
            return outs;
        }
        var bpr = K >> 8;

        var pooled = sp != null;
        var qs = pooled ? sp.scratchBytes(0, K) : Bytes.alloc(K);
        var bsums = pooled ? sp.scratchBytes(1, bpr * 16 * 4) : Bytes.alloc(bpr * 16 * 4);
        var dScales = pooled ? sp.scratchBytes(2, bpr * 4) : Bytes.alloc(bpr * 4);
        var aBase = qs.address();
        var dBase = dScales.address();
        var xBase = x.data().raw();
        var _prof = sp != null && sp.profiling();
        var _tq = _prof ? Sys.time() : 0.0;
        quantizeAll(xBase, aBase, bsums, dBase, bpr, sp);
        if (_prof) sp.addQuantUs(Std.int((Sys.time() - _tq) * 1e6));

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

        var y0 = Tensor.uninit([1, r0], DType.F32);
        var y1 = Tensor.uninit([1, r1], DType.F32);
        var y2 = (w2 != null) ? Tensor.uninit([1, r2], DType.F32) : null;
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
                    var xdb = Mem.loadF32(dBase + Usize.fromInt(b << 2));
                    sum += isQ6
                        ? q6DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb)
                        : q4DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsums, b * 16, xdb);
                }
                Mem.storeF32(yb + Usize.fromInt(n << 2), sum);
            }
        };
        if (sp != null) sp.parallelRows(total, band);
        else band(0, total, 0);

        if (!pooled) {
            qs.free();
            bsums.free();
            dScales.free();
        }
        var outs = [y0, y1];
        if (w2 != null) outs.push(y2);
        return outs;
    }

    static function runBandedFused(w0:QTensor, w1:QTensor, w2:QTensor,
            batch:Int, K:Int, aBase:Usize, bsums:Bytes, dBase:Usize,
            sp:Null<SpinPool>):Array<Tensor> {
        var bpr = K >> 8;
        var r0 = w0.rows();
        var r1 = w1.rows();
        var r2 = (w2 != null) ? w2.rows() : 0;
        var e0 = r0;
        var e1 = r0 + r1;
        var total = r0 + r1 + r2;

        var wb0 = w0.dataPtr();
        var wb1 = w1.dataPtr();
        var wb2 = (w2 != null) ? w2.dataPtr() : wb0;
        var q60 = (w0.scheme() == QScheme.Q6_K);
        var q61 = (w1.scheme() == QScheme.Q6_K);
        var q62 = (w2 != null) && (w2.scheme() == QScheme.Q6_K);

        var y0 = Tensor.uninit([batch, r0], DType.F32);
        var y1 = Tensor.uninit([batch, r1], DType.F32);
        var y2 = (w2 != null) ? Tensor.uninit([batch, r2], DType.F32) : null;
        var yb0 = y0.data().raw();
        var yb1 = y1.data().raw();
        var yb2 = (w2 != null) ? y2.data().raw() : yb0;

        var band = function(n0:Int, n1:Int, node:Int):Void {
            for (g in n0...n1) {
                var wBase:Usize; var isQ6:Bool; var n:Int; var yb:Usize; var outRows:Int;
                if (g < e0) {
                    wBase = wb0; isQ6 = q60; n = g; yb = yb0; outRows = r0;
                } else if (g < e1) {
                    wBase = wb1; isQ6 = q61; n = g - e0; yb = yb1; outRows = r1;
                } else {
                    wBase = wb2; isQ6 = q62; n = g - e1; yb = yb2; outRows = r2;
                }
                var blockBytes = isQ6 ? 210 : 144;
                if (batch == 1) {
                    var sum = 0.0;
                    var base = n * bpr;
                    for (b in 0...bpr) {
                        var xdb = Mem.loadF32(dBase + Usize.fromInt(b << 2));
                        sum += isQ6
                            ? q6DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsums, b * 16, xdb)
                            : q4DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsums, b * 16, xdb);
                    }
                    Mem.storeF32(yb + Usize.fromInt(n << 2), sum);
                    continue;
                }
                var rt = 0;
                while (rt + 4 <= batch) {
                    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
                    for (b in 0...bpr) {
                        var blkOff = (n * bpr + b) * blockBytes;
                        var a0 = rt * K + b * 256;
                        var i0 = (rt * bpr + b) * 16;
                        var d0 = Mem.loadF32(dBase + Usize.fromInt((rt * bpr + b) << 2));
                        var d1 = Mem.loadF32(dBase + Usize.fromInt(((rt + 1) * bpr + b) << 2));
                        var d2 = Mem.loadF32(dBase + Usize.fromInt(((rt + 2) * bpr + b) << 2));
                        var d3 = Mem.loadF32(dBase + Usize.fromInt(((rt + 3) * bpr + b) << 2));
                        if (isQ6) {
                            s0 += q6DotMA4(wBase, blkOff, aBase, a0, bsums, i0, d0);
                            s1 += q6DotMA4(wBase, blkOff, aBase, a0 + K, bsums, i0 + bpr * 16, d1);
                            s2 += q6DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsums, i0 + 2 * bpr * 16, d2);
                            s3 += q6DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsums, i0 + 3 * bpr * 16, d3);
                        } else {
                            s0 += q4DotMA4(wBase, blkOff, aBase, a0, bsums, i0, d0);
                            s1 += q4DotMA4(wBase, blkOff, aBase, a0 + K, bsums, i0 + bpr * 16, d1);
                            s2 += q4DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsums, i0 + 2 * bpr * 16, d2);
                            s3 += q4DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsums, i0 + 3 * bpr * 16, d3);
                        }
                    }
                    Mem.storeF32(yb + Usize.fromInt((rt * outRows + n) << 2), s0);
                    Mem.storeF32(yb + Usize.fromInt(((rt + 1) * outRows + n) << 2), s1);
                    Mem.storeF32(yb + Usize.fromInt(((rt + 2) * outRows + n) << 2), s2);
                    Mem.storeF32(yb + Usize.fromInt(((rt + 3) * outRows + n) << 2), s3);
                    rt += 4;
                }
                while (rt < batch) {
                    var sum = 0.0;
                    for (b in 0...bpr) {
                        var blkOff = (n * bpr + b) * blockBytes;
                        var xdb = Mem.loadF32(dBase + Usize.fromInt((rt * bpr + b) << 2));
                        sum += isQ6
                            ? q6DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsums, (rt * bpr + b) * 16, xdb)
                            : q4DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsums, (rt * bpr + b) * 16, xdb);
                    }
                    Mem.storeF32(yb + Usize.fromInt((rt * outRows + n) << 2), sum);
                    rt++;
                }
            }
        };
        if (sp != null) sp.parallelRows(total, band);
        else band(0, total, 0);

        var outs = [y0, y1];
        if (w2 != null) outs.push(y2);
        return outs;
    }
}
