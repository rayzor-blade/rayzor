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
    static var _amx:Int = 0;
    /** Minimum prefill batch (= prompt tokens) worth handing to AMX.
        MEASURED CROSSOVER, interleaved arms with cooldowns, prefill timed
        directly on Qwen Q5_K_M (prefill_fwd_s, median of 2):
            16 tok  +61.2%   AMX much slower
            39 tok   +4.8%   AMX slower
            69 tok   +7.8%   AMX slower
           129 tok   -0.5%   tie  <- crossover
           249 tok   -5.9%   AMX faster
           429 tok   -4.4%   AMX faster
        The old value of 16 admitted work 8x too small to amortise
        dequant-to-F32 + sgemm setup — a 16-token prompt sat exactly on the
        threshold and paid 61%. AMX itself is sound platform leverage; the GATE
        was wrong. `NUE_AMX_MIN_BATCH` overrides (other silicon will differ). */
    static var _amxMin:Int = 0;
    public static function amxMinBatch():Int {
        if (_amxMin == 0) {
            var v = Sys.getEnvOr("NUE_AMX_MIN_BATCH", "RZT_AMX_MIN_BATCH");
            var n = (v != null && v != "") ? Std.int(Std.parseFloat(v)) : 0;
            _amxMin = (n > 0) ? n : 128;
        }
        return _amxMin;
    }

    /** Mac AMX prefill routing — ON by default (RZT_AMX_PREFILL=0 opts out):
        compute-bound Q4 batches hand off to the runtime kernel's Accelerate
        f16 GEMM (dequant-once weights). Measured: batch=497 prefill ttft
        7.0s vs 9.4s SDOT. Decode and non-Mac are never routed. */
    static function amxPrefill():Bool {
        if (_amx == 0) {
            var v = Sys.getEnvOr("RZT_AMX_PREFILL", "RAYZOR_AMX_PREFILL");
            var off = (v != null && (v == "0" || v == "false"));
            var on = !off && (Sys.systemName() == "Mac");
            _amx = on ? 1 : 2;
        }
        return _amx == 1;
    }

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

    /**
     * `NUE_HAXE_Q8_0=0` routes Q8_0 back to the runtime XTQ kernel.
     *
     * Exists so the Haxe Q8_0 kernel can be A/B'd against the Rust one with the
     * SAME BINARY and the SAME flags — one variable, with the census printing
     * which path actually ran. Measuring it by editing dispatch and rebuilding
     * changes the compile as well as the kernel, and that is how a three-way
     * change got mistaken for a kernel result.
     *
     * Defaults ON. Rested ABBA at 512 tok on Qwen Q6_K (3 rounds, load-gated,
     * census-verified arms, same binary): Haxe median 102.6 tok/s vs Rust
     * 82.3 — the pure-Haxe stack (this kernel + the capture-fixed Q6_K band)
     * BEATS the runtime kernel by ~25%. The kernel was 3-4x SLOWER until the
     * band closure stopped capturing refcounted Bytes; the arithmetic never
     * was the problem (scale-load + SIMD4f fold + row blocking: ~8% combined;
     * capture-set fix: 2.5x). `NUE_HAXE_Q8_0=0` returns to the Rust kernel
     * for A/B.
     *
     * Cached like useHaxeMatmul (0 = uninitialised, load-bearing for x-module
     * static dups).
     */
    static var _haxeQ8_0:Int = 0;

    static function useHaxeQ8_0():Bool {
        if (_haxeQ8_0 == 0) {
            var v = Sys.getEnvOr("NUE_HAXE_Q8_0", "RAYZOR_HAXE_Q8_0");
            _haxeQ8_0 = (v != null && (v == "0" || v == "false")) ? 2 : 1;
        }
        return _haxeQ8_0 == 1;
    }

    // Same A/B gate for the pure-Haxe INT8 kernel. Defaults ON;
    // `NUE_HAXE_INT8=0` returns INT8 matmuls to the runtime XTQ kernel.
    static var _haxeInt8:Int = 0;

    static function useHaxeInt8():Bool {
        if (_haxeInt8 == 0) {
            var v = Sys.getEnvOr("NUE_HAXE_INT8", "RAYZOR_HAXE_INT8");
            _haxeInt8 = (v != null && (v == "0" || v == "false")) ? 2 : 1;
        }
        return _haxeInt8 == 1;
    }

    // ---- dispatch census -------------------------------------------------
    // `NUE_DUMP_Q4_GATES=1` prints, per QScheme, how many matmuls went to a
    // Haxe kernel vs escaped to the Rust FFI. "Is this run actually pure
    // Haxe?" must be a PRINTED FACT: inferring it from dispatch source and
    // perf deltas is how a Q8_0 lm_head kept running Rust on every token
    // inside a run labelled pure-Haxe. A parity claim requires ffi=0.
    // Index = QScheme ordinal (0 INT8, 1 Q4_K_M, 2 Q6_K, 3 Q8_0).
    static var _censusHaxe:Array<Int> = [0, 0, 0, 0];
    static var _censusFfi:Array<Int> = [0, 0, 0, 0];
    // Escapes to a PLATFORM API (Apple AMX/Accelerate) are counted apart from
    // escapes to a Rust KERNEL. Only the latter is a parity failure: platform
    // APIs are the FFI the pure-Haxe direction sanctions, so a correct AMX
    // prefill must not read as "NOT pure Haxe" — that sends the next reader
    // chasing a non-bug (it did).
    static var _censusPlatform:Array<Int> = [0, 0, 0, 0];
    static var _censusOn:Int = 0;

    static function censusEnabled():Bool {
        if (_censusOn == 0) {
            var v = Sys.getEnvOr("NUE_DUMP_Q4_GATES", "RAYZOR_DUMP_Q4_GATES");
            _censusOn = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _censusOn == 1;
    }

    /** Escape to a platform API (AMX/Accelerate) — sanctioned, not a parity
        failure. */
    static inline function censusPlatform(schemeOrd:Int):Void {
        if (censusEnabled() && schemeOrd >= 0 && schemeOrd < 4) {
            _censusPlatform[schemeOrd] = _censusPlatform[schemeOrd] + 1;
        }
    }

    static inline function census(schemeOrd:Int, ffi:Bool):Void {
        if (censusEnabled() && schemeOrd >= 0 && schemeOrd < 4) {
            if (ffi) _censusFfi[schemeOrd] = _censusFfi[schemeOrd] + 1;
            else _censusHaxe[schemeOrd] = _censusHaxe[schemeOrd] + 1;
        }
    }

    // ---- plan census (NueGraph Stage 0: make the implicit plan printable) --
    // The kernel/fusion/dispatch decisions are currently spread across gates in
    // Q4Matmul, GQAttention and SwiGLU, so "what plan did this run actually
    // execute?" was unanswerable without reading source and guessing. These
    // count what the dispatcher DID; `dumpPlan` prints it next to the gates
    // that were in force. A stale gate (NUE_FUSED_MATMUL sat off for months on
    // a refuted verdict) shows up here as fused=0 at a glance.
    static var _planFusedTriple:Int = 0;
    static var _planSplitTriple:Int = 0;
    static var _planFusedPair:Int = 0;
    static var _planSplitPair:Int = 0;
    static var _planOneDispatch:Int = 0;
    static var _planPerWeight:Int = 0;

    static inline function planFused(isTriple:Bool, oneDispatch:Bool):Void {
        if (censusEnabled()) {
            if (isTriple) _planFusedTriple = _planFusedTriple + 1;
            else _planFusedPair = _planFusedPair + 1;
            if (oneDispatch) _planOneDispatch = _planOneDispatch + 1;
            else _planPerWeight = _planPerWeight + 1;
        }
    }

    static inline function planSplit(isTriple:Bool):Void {
        if (censusEnabled()) {
            if (isTriple) _planSplitTriple = _planSplitTriple + 1;
            else _planSplitPair = _planSplitPair + 1;
        }
    }

    static inline function onOff(b:Bool):String return b ? "on" : "off";

    /** Print the EFFECTIVE PLAN: the gates in force and what the dispatcher
        actually fused/split. Safe to call when the census is off.

        NO PARAMETERS ON PURPOSE. A public static taking a cross-module type
        (e.g. `SpinPool`) perturbs module registration enough to fail its own
        registration (`E0100 class not registered`) and to break unrelated
        cross-module field access — the same trap already documented on
        `matmulQ8_0`'s explicit-`sp` note. Callers print the pool line
        themselves from `spinPool.profReport()`. This is a workaround for the
        x-module static-resolution cluster, not a design choice. */
    public static function dumpPlan():Void {
        if (!censusEnabled()) return;
        // NB: the haxe_matmul gate itself lives on Linear; reading it here
        // would be a cross-module STATIC access, which the x-module cluster
        // does not forward (it fails this class's own registration, E0100).
        // If this line prints at all, Linear routed us here, so haxe_matmul
        // was on.
        Sys.println("[nue-plan] gates: haxe_matmul=on(implied)"
            + " int8=" + onOff(useHaxeInt8())
            + " q8_0=" + onOff(useHaxeQ8_0())
            + " rowwise_fusion=" + onOff(useRowwiseFusion())
            + " fused_dispatch=" + onOff(useFusedDispatch())
            + " kquant_fusion=" + onOff(useFusedMatmul())
            + " amx_prefill=" + onOff(amxPrefill()));
        Sys.println("[nue-plan] fusion: triples fused=" + _planFusedTriple
            + " split=" + _planSplitTriple
            + " | pairs fused=" + _planFusedPair
            + " split=" + _planSplitPair
            + " | banding one-dispatch=" + _planOneDispatch
            + " per-weight=" + _planPerWeight);
        Sys.println("[nue-plan] pool: workers=" + workerCount()
            + " profile=" + poolProfile()
            + " spins=" + poolSpins()
            + "  (band/quant/dispatch counters: NUE_PROFILE_POOL=1)");
        dumpCensus();
    }

    /** Print the census. Callers dump this at exit; safe to call when the
        census is off (prints nothing). */
    public static function dumpCensus():Void {
        if (!censusEnabled()) return;
        var names = ["INT8", "Q4_K_M", "Q6_K", "Q8_0"];
        var totalFfi = 0;
        var totalPlat = 0;
        for (i in 0...4) totalFfi += _censusFfi[i];
        for (i in 0...4) totalPlat += _censusPlatform[i];
        for (i in 0...4) {
            if (_censusHaxe[i] > 0 || _censusFfi[i] > 0 || _censusPlatform[i] > 0) {
                Sys.println("[q4-census] " + names[i] + " haxe=" + _censusHaxe[i]
                    + " kernel_ffi=" + _censusFfi[i]
                    + " platform=" + _censusPlatform[i]);
            }
        }
        // The parity verdict keys on KERNEL escapes only. Platform escapes are
        // reported so an AMX run is never mistaken for a pure-Haxe one, but
        // they do not fail the claim.
        Sys.println("[q4-census] total kernel_ffi=" + totalFfi
            + " platform=" + totalPlat
            + (totalFfi == 0
                ? (totalPlat == 0 ? "  (PURE HAXE)" : "  (PURE HAXE kernels + platform API)")
                : "  (NOT pure Haxe — escaped to a Rust kernel)"));
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
    static inline function q4DotMA4(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsAddr:Usize, bsBase:Int, xd:Float):Float {
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
        var imin = mn0 * (Mem.loadI32(bsAddr + Usize.fromInt(b4)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 4)))
                 + mn1 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 8)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 12)))
                 + mn2 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 16)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 20)))
                 + mn3 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 24)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 28)))
                 + mn4 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 32)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 36)))
                 + mn5 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 40)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 44)))
                 + mn6 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 48)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 52)))
                 + mn7 * (Mem.loadI32(bsAddr + Usize.fromInt(b4 + 56)) + Mem.loadI32(bsAddr + Usize.fromInt(b4 + 60)));
        return xd * (d * isum.sum() - dmin * imin);
    }

    /** One Q6_K super-block's f32 contribution (256 weights, 210-byte block).
        Rebuilds each 6-bit weight from its ql low-nibble + qh 2-bit pair and
        dots it against the shared Q8_K activation with SDOT; the unsigned
        0..63 → −32..31 bias folds into 32·Σx via the per-16 bsums. Mirrors
        runtime-core `vec_dot_q6_K_q8_K`. Safe to call concurrently across
        output-row bands (no shared scratch). */
    /** Signed-i8 scale byte at `wBase+off`, matching SIMD16i8.get()'s
        sign-extend contract. Reading the Q6_K sub-block scale via a loaded
        `SIMD16i8.get(dynamic-lane)` lowered to a 16-way extract+select chain
        (SIMD16i8_extract); a direct memory byte-load is bit-identical and skips
        the scalarization. */
    static inline function scaleI8(wBase:Usize, off:Int):Int {
        var b = Mem.loadU8(wBase + Usize.fromInt(off));
        return (b << 24) >> 24;
    }

    static inline function q6DotMA4(wBase:Usize, wBlk:Int, aBase:Usize, aBlk:Int, bsAddr:Usize, bsBase:Int, xd:Float):Float {
        // scales are read per-(n,j) with a dynamic index — take them straight
        // from memory via scaleI8 (see there); dVec keeps its constant-lane gets.
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
                var dScLo:Float = d * scaleI8(wBase, wBlk + 192 + scOff + 2 * j);
                var dScHi:Float = d * scaleI8(wBase, wBlk + 192 + scOff + 2 * j + 1);
                sumTerm1 += dScLo * sdotLo + dScHi * sdotHi;
                var bi = bsBase + ((outOff + j * 32) >> 4);
                sumTerm2 += 32.0 * dScLo * Mem.loadI32(bsAddr + Usize.fromInt(bi * 4))
                          + 32.0 * dScHi * Mem.loadI32(bsAddr + Usize.fromInt((bi + 1)) * 4);
            }
        }
        return xd * (sumTerm1 - sumTerm2);
    }

    // Matmul pool policy — Nue owns it; the SpinPool primitive is env-agnostic
    // and receives the resolved config. Profiles trade peak for co-runner
    // robustness: throughput (all P-cores, default), cooperative (P-1, gentle
    // spin so idle workers cede cores), latency (P-1, minimal spin).
    //
    // Resolution order: NUE_POOL_PROFILE env (explicit runtime choice) wins,
    // then a code-set override (an app that knows its own scenario — e.g. a
    // decode loop that also runs a stream-writer thread and wants to leave the
    // P-cluster a free core), then the throughput default.
    static var _profileOverride:String = null;

    /** Set the pool profile from code. `null` clears it. NUE_POOL_PROFILE still
        takes precedence, so this is a scenario default, not a hard override. */
    public static function setPoolProfile(profile:String):Void {
        _profileOverride = profile;
    }

    public static function poolProfile():String {
        var p = Sys.getEnvOr("NUE_POOL_PROFILE", "RAYZOR_HAXE_POOL_PROFILE");
        if (p != null && p != "") return p;
        if (_profileOverride != null && _profileOverride != "") return _profileOverride;
        return "throughput";
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

    /** Adaptive (wall-clock) tight-spin bound; `NUE_POOL_ADAPTIVE=0` reverts to
        the iteration-only bound. Prime suspect for the occasional decode stall
        — see PERFORMANCE.md. */
    public static function poolAdaptive():Bool {
        var env = Sys.getEnvOr("NUE_POOL_ADAPTIVE", "RAYZOR_HAXE_POOL_ADAPTIVE");
        return !(env == "0" || env == "false");
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
    /** True for schemes whose storage is NOT a 256-element k-quant super-block
        (per-row INT8, or native Q8_0's 32-element blocks) — those must never
        reach the banded k-quant loops, which would misread the layout. */
    static inline function isRowwise(qw:QTensor):Bool {
        return qw.scheme() == QScheme.INT8 || qw.scheme() == QScheme.Q8_0;
    }

    /** Row-wise (INT8/Q8_0) triple fusion. Default ON — it is a DIFFERENT
        mechanism from `useFusedMatmul()`'s k-quant row-space fusion (which
        stays opt-in because it regressed on macOS): here each weight keeps its
        own band kernel and only the ACTIVATION QUANTISE is shared, so output is
        bit-identical to the split path and the win is pure removed work
        (measured Qwen Q5_0: 73.65 -> 93.81 tok/s, and 107.00 with
        NUE_POOL_SPINS=100). `NUE_FUSED_ROWWISE=0` disables. */
    static var _fusedRowwise:Int = 0;
    public static function useRowwiseFusion():Bool {
        if (_fusedRowwise == 0) {
            var v = Sys.getEnvOr("NUE_FUSED_ROWWISE", "RAYZOR_HAXE_FUSED_ROWWISE");
            _fusedRowwise = (v != null && (v == "0" || v == "false")) ? 2 : 1;
        }
        return _fusedRowwise == 1;
    }

    /** INT8 weight whose Haxe band kernel is live. */
    static inline function int8Bandable(qw:QTensor):Bool {
        return qw.scheme() == QScheme.INT8 && useHaxeInt8();
    }

    /** Single-dispatch banding over a fused triple's concatenated row space
        (`NUE_FUSED_DISPATCH=0` falls back to one dispatch per weight, for A/B).
        Default ON. */
    static var _fusedDispatch:Int = 0;
    public static function useFusedDispatch():Bool {
        if (_fusedDispatch == 0) {
            var v = Sys.getEnvOr("NUE_FUSED_DISPATCH", "RAYZOR_HAXE_FUSED_DISPATCH");
            _fusedDispatch = (v != null && (v == "0" || v == "false")) ? 2 : 1;
        }
        return _fusedDispatch == 1;
    }

    /** True when this triple would take the fused row-wise path — call sites use
        it to decide whether to route through `matmulFused` at all.

        DEFAULT-ON only for ALL-INT8 triples: interleaved A/B (the only kind that
        survives this machine's drift) measured Q5_0 +25.0% with NON-OVERLAPPING
        ranges (ON 100.7-102.8 vs OFF 80.7-82.8), but Q6_K's Q8_0 pairs came in
        at -2.3% with ranges almost entirely overlapping — no win there, so Q8_0
        triples keep their previous behaviour and only fuse under the explicit
        `useFusedMatmul()` opt-in. */
    public static function canFuseRowwise(w0:QTensor, w1:QTensor, w2:QTensor):Bool {
        if (w0 == null || w1 == null) return false;
        var allRow = rowwiseHaxeBandable(w0) && rowwiseHaxeBandable(w1)
            && (w2 == null || rowwiseHaxeBandable(w2));
        if (!allRow) return false;
        if (useFusedMatmul()) return true; // opt-in: any row-wise mix
        // Row-wise triples of ANY mix (INT8 and/or Q8_0) — each weight keeps
        // its own row kernel and the joined row space is banded in one
        // dispatch. Previously INT8-only, because Q8_0 fusion measured
        // neutral-to-negative when it still dispatched PER WEIGHT; with the
        // single dispatch that objection no longer applies (see PERFORMANCE.md).
        return useRowwiseFusion();
    }

    /** Row-wise AND its pure-Haxe band kernel is enabled — i.e. this weight can
        join a fused triple over one shared per-row int8 activation. A weight
        whose kernel is gated off must NOT be fused: it has to reach `matmul` so
        it can take the runtime XTQ bail. One `scheme()` read (it is an opaque
        cross-dylib call the optimiser cannot CSE). */
    static inline function rowwiseHaxeBandable(qw:QTensor):Bool {
        var s = qw.scheme();
        return (s == QScheme.INT8) ? useHaxeInt8() : ((s == QScheme.Q8_0) ? useHaxeQ8_0() : false);
    }

    /** Band one row-wise weight against an ALREADY-quantised shared activation,
        picking the kernel that matches its weight layout. Activation encoding is
        identical for both; only the weight-side scale differs. */
    static inline function rowwiseBand(qw:QTensor, batch:Int, K:Int, aBase:Usize,
            sBase:Usize, sp:Null<SpinPool>):Tensor {
        if (qw.scheme() == QScheme.INT8) {
            census(0, false);
            return int8Banded(qw, batch, K, aBase, sBase, sp);
        }
        census(3, false);
        return q8Banded(qw, batch, K, aBase, sBase, sp);
    }

    public static function matmul(qw:QTensor, x:Tensor, ?sp:SpinPool):Tensor {
        // Q8_0 has its own pure-Haxe kernel: 32-element blocks with a per-block
        // f16 scale, which the 256-wide k-quant band loop below would misread.
        if (qw.scheme() == QScheme.Q8_0) {
            if (useHaxeQ8_0()) {
                census(3, false);
                return matmulQ8_0(qw, x, sp);
            }
            census(3, true);
            return qw.matmulXTQThreaded(x, 0);
        }
        // INT8 per-row weights (Q5_0/Q5_1-sourced): pure-Haxe band kernel,
        // same shape as matmulQ8_0 but one f32 scale per ROW (read from
        // qw.scalesPtr()) instead of a per-block f16 scale. `NUE_HAXE_INT8=0`
        // returns to the runtime XTQ kernel for A/B.
        if (qw.scheme() == QScheme.INT8) {
            if (useHaxeInt8()) {
                census(0, false);
                return matmulInt8(qw, x, sp);
            }
            census(0, true);
            return qw.matmulXTQThreaded(x, 0);
        }
        var K = qw.cols();
        var bpr = K >> 8;
        var batch = Std.int(x.numel() / K);
        if (batch < 1) batch = 1;

        // Opt-in AMX prefill experiment: compute-bound Q4_K_M batches hand off
        // to the runtime kernel (Accelerate dequant-once + sgemm). Q4_K_M ONLY:
        // the runtime kernel's AMX and morsel paths both require Q4_K_M, so a
        // routed Q6_K falls to its legacy per-block fallback (~700ms/call at
        // batch=497 — measured, the whole "AMX long-prompt bleed"). Q6_K and
        // decode stay on the Haxe band loop below.
        if (batch >= amxMinBatch() && amxPrefill() && qw.scheme() == QScheme.Q4_K_M) {
            // PLATFORM escape, not a kernel escape: Apple's AMX/Accelerate
            // only exists across the FFI boundary. Counted separately so an
            // AMX prefill never reads as "NOT pure Haxe" — platform APIs are
            // the FFI this direction sanctions; Rust kernels are not.
            censusPlatform(1);
            return qw.matmulXTQThreaded(x, 0);
        }

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

        census(qw.scheme() == QScheme.Q6_K ? 2 : 1, false);
        var y = runBanded(qw, batch, K, aBase, bsums, dBase, sp);

        if (!pooled) {
            qs.free();
            bsums.free();
            dScales.free();
        }
        return y;
    }

    // ---- Q8_0 -----------------------------------------------------------
    // Q8_0 block = 34 bytes: little-endian f16 scale at [0..1], then 32
    // signed int8 weights at [2..33]. No super-block and no mins — the
    // simplest layout we carry, and the one Qwen's 152k-vocab lm_head uses,
    // so it runs on EVERY decoded token.
    //
    // Activations are quantised per ROW (one scale for the whole row), not
    // per 256-element super-block like Q8_K, so this path owns its own
    // quantiser and cannot share the banded k-quant machinery above.
    static inline var Q8_0_BLOCK_BYTES:Int = 34;
    static inline var Q8_0_BLOCK_ELEMS:Int = 32;

    /**
     * Symmetric per-row int8 quantise: `scale = max|x| / 127`, round half
     * away from zero, clamp to +/-127. Writes `K` int8 bytes at `aBase` and
     * returns the row scale (0.0 for an all-zero row, whose bytes are zeroed).
     *
     * Deliberately branch-free in the inner loop via ternaries: an
     * `if (c) f = expr` on a loop-carried Float next to Math.floor/ceil boxes
     * per element here (see bugs_float_conditional_reassign_boxes), which on
     * a 151936-row lm_head is a per-token allocation storm.
     */
    static function quantizeRowI8(xBase:Usize, aBase:Usize, K:Int):Float {
        // SIMD maxAbs: 4-wide abs+max, 2 accumulators (breaks the loop-carried
        // max latency chain), scalar tail for K not a multiple of 8. IEEE max
        // is exact + associative and abs(v)==max(v,-v) for non-NaN activations,
        // so this is bit-identical to the scalar reduction. (The quantise pass
        // below stays scalar — a SIMD f32->i8 narrow would need new round +
        // pack primitives; the maxAbs pass needs none.)
        var m0 = SIMD4f.splat(0.0);
        var m1 = SIMD4f.splat(0.0);
        var i = 0;
        while (i + 8 <= K) {
            var v0 = SIMD4f.load(Ptr.fromRaw(xBase + Usize.fromInt(i << 2)));
            var v1 = SIMD4f.load(Ptr.fromRaw(xBase + Usize.fromInt((i + 4) << 2)));
            m0 = m0.max(v0.abs());
            m1 = m1.max(v1.abs());
            i += 8;
        }
        var mm = m0.max(m1);
        var l0 = mm.get(0);
        var l1 = mm.get(1);
        var l2 = mm.get(2);
        var l3 = mm.get(3);
        var mx01 = l0 > l1 ? l0 : l1;
        var mx23 = l2 > l3 ? l2 : l3;
        var maxAbs = mx01 > mx23 ? mx01 : mx23;
        while (i < K) {
            var v = Mem.loadF32(xBase + Usize.fromInt(i << 2));
            var a = (v < 0.0) ? -v : v;
            maxAbs = (a > maxAbs) ? a : maxAbs;
            i++;
        }
        if (maxAbs == 0.0) {
            for (i in 0...K) Mem.storeU8(aBase + Usize.fromInt(i), 0);
            return 0.0;
        }
        var inv = 127.0 / maxAbs;
        for (i in 0...K) {
            var v = Mem.loadF32(xBase + Usize.fromInt(i << 2)) * inv;
            // Round-half-away via Std.int (fptosi intrinsic, no box) — the
            // same idiom quantizeBlock uses. Math.floor/ceil boxes the Float
            // argument per element, which on a 151936-row lm_head is millions
            // of heap allocs per token. Clamp to +/-127 (not -128): the
            // reference kernel keeps the activation range sign-symmetric.
            var q = v >= 0 ? Std.int(v + 0.5) : Std.int(v - 0.5);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            Mem.storeU8(aBase + Usize.fromInt(i), q & 0xFF);
        }
        return maxAbs / 127.0;
    }

    /** One 32-element Q8_0 weight block against 32 int8 activations -> i32.
        Two 16-lane SDOT chains; `dot` (signed x signed), NOT `dotI8I7` —
        Q8_0 weights use the full i8 range, which the i7 variant forbids. */
    static inline function q8DotBlock(wBase:Usize, wOff:Int, aBase:Usize, aOff:Int):Int {
        // Distinct bindings per step, never a reassigned accumulator: passing
        // a SIMD value as an argument MOVES it, so `acc = dot(acc, ..)`
        // followed by `acc.sum()` is a use-after-move. Chaining keeps one
        // horizontal sum for the whole 32-element block.
        var acc0 = SIMD4i32.dot(SIMD4i32.splat(0),
            SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(aOff))),
            SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wOff + 2))));
        var acc1 = SIMD4i32.dot(acc0,
            SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(aOff + 16))),
            SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wOff + 2 + 16))));
        return acc1.sum();
    }

    /**
     * `y = x @ qw.T` for a Q8_0 weight, in pure Haxe.
     *
     * Replaces the `matmulXTQThreaded` FFI bail-out this path used to take:
     * the band loop below is 256-wide k-quant machinery that would misread
     * Q8_0's 32-element blocks, so the scheme escaped to Rust rather than
     * getting a kernel. Row scales fold in once at the end of each row —
     * `y[b,n] = xScale[b] * SUM_blk wScale[blk] * dot_i32(...)`.
     */
    // Private + explicit `sp` (no `?optional`): a public static with a
    // trailing optional arg perturbed module registration enough to break an
    // unrelated cross-module field access (LlamaArch's `model.spinPool`,
    // E0100) — the x-module resolution cluster. Only `matmul` calls this.
    static function matmulQ8_0(qw:QTensor, x:Tensor, sp:Null<SpinPool>):Tensor {
        var K = qw.cols();
        var batch = Std.int(x.numel() / K);
        if (batch < 1) batch = 1;

        var pooled = sp != null;
        var qs = pooled ? sp.scratchBytes(0, batch * K) : Bytes.alloc(batch * K);
        var aBase = qs.address();
        var xScales = pooled ? sp.scratchBytes(2, batch * 4) : Bytes.alloc(batch * 4);
        var sBase = xScales.address();
        // Time the shared activation quantise like the k-quant path does —
        // without this the row-wise kernels report quant_ms=0 and the split
        // between quantise and band is invisible.
        var _prof = sp != null && sp.profiling();
        var _tq = _prof ? Sys.time() : 0.0;
        quantiseRowsI8(x.data().raw(), aBase, sBase, batch, K);
        if (_prof) sp.addQuantUs(Std.int((Sys.time() - _tq) * 1e6));

        var y = q8Banded(qw, batch, K, aBase, sBase, sp);
        if (!pooled) {
            qs.free();
            xScales.free();
        }
        return y;
    }

    /** Symmetric int8-quantise `batch` activation rows into `aBase`, one f32
        scale per row into `sBase`. Split out of matmulQ8_0 so a fused call can
        quantise a SHARED activation exactly once instead of once per weight. */
    static function quantiseRowsI8(xBase:Usize, aBase:Usize, sBase:Usize,
            batch:Int, K:Int):Void {
        for (b in 0...batch) {
            var s = quantizeRowI8(xBase + Usize.fromInt(b * K * 4),
                aBase + Usize.fromInt(b * K), K);
            Mem.storeF32(sBase + Usize.fromInt(b << 2), s);
        }
    }

    /** Band one Q8_0 weight against an ALREADY-quantised activation. */
    static function q8Banded(qw:QTensor, batch:Int, K:Int, aBase:Usize,
            sBase:Usize, sp:Null<SpinPool>):Tensor {
        var rows = qw.rows();
        var blocks = Std.int(K / Q8_0_BLOCK_ELEMS);
        var wBase = qw.dataPtr();
        var y = Tensor.uninit([batch, rows], DType.F32);
        var yBase = y.data().raw();
        // Hoisted to locals: the band closure must capture plain values, not
        // reference class-level `static inline var`s (that shape broke an
        // unrelated cross-module field access — E0100 on LlamaModel.spinPool).
        var blockBytes = Q8_0_BLOCK_BYTES;
        var blockElems = Q8_0_BLOCK_ELEMS;

        var band = function(n0:Int, n1:Int, node:Int):Void {
            var rowBytes = blocks * blockBytes;
            for (b in 0...batch) {
                var aRow = b * K;
                var xs = Mem.loadF32(sBase + Usize.fromInt(b << 2));
                var n = n0;
                // FOUR output rows per step: independent chains, and the shared
                // activation row stays L1-hot across all four.
                while (n + 4 <= n1) {
                    var s0 = q8RowDot(wBase, n * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                    var s1 = q8RowDot(wBase, (n + 1) * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                    var s2 = q8RowDot(wBase, (n + 2) * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                    var s3 = q8RowDot(wBase, (n + 3) * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                    var o = (b * rows + n) << 2;
                    Mem.storeF32(yBase + Usize.fromInt(o), s0 * xs);
                    Mem.storeF32(yBase + Usize.fromInt(o + 4), s1 * xs);
                    Mem.storeF32(yBase + Usize.fromInt(o + 8), s2 * xs);
                    Mem.storeF32(yBase + Usize.fromInt(o + 12), s3 * xs);
                    n += 4;
                }
                while (n < n1) {
                    Mem.storeF32(yBase + Usize.fromInt((b * rows + n) << 2),
                        q8RowDot(wBase, n * rowBytes, aBase, aRow, blocks, blockBytes, blockElems) * xs);
                    n++;
                }
            }
        };
        if (sp != null) sp.parallelRows(rows, band);
        else band(0, rows, 0);
        return y;
    }

    // ---- INT8 (per-row symmetric) --------------------------------------
    // Weight = packed int8, `numel = rows*K`, ONE f32 scale per row in the
    // separate `meta` array (qw.scalesPtr()). No blocks, no inline scale, no
    // f16 decode — simpler than Q8_0. Activations quantise per row exactly as
    // Q8_0 (shared quantiseRowsI8), so:
    //   y[b,n] = xScale[b] * wScale[n] * SUM_k a_i8[b,k] * w_i8[n,k]

    /** SDOT one INT8 weight row (`K` int8 at wBase+wOff) against `K` int8
        activations (aBase+aOff) -> i32. FOUR independent loop-carried SIMD4i32
        accumulators over 16-lane chunks, hsum'd once at the end. SDOT has ~5-cycle
        latency, so a SINGLE dependent chain stalls on it (measured 45 vs 64 tok/s
        for one accumulator); four independent chains give the scheduler four
        in-flight SDOTs to hide the latency, while still paying only 4 hsums per
        row instead of one per chunk. The loop-carried `acc = SIMD4i32.dot(acc,..)`
        relies on the SIMD-vector-phi fix (49ab3808). `dot` (signed x signed):
        INT8 uses the full i8 range, forbidden to the i7 variant. 16-chunk +
        scalar tail for K not a multiple of 64.

        MEASURED vs runtime-core's `dot_i8_i8` SHAPE (2 accumulators, step 32,
        two hsums): that form is 5.6% SLOWER here (interleaved A/B medians
        90.25 vs 95.30 tok/s, non-overlapping ranges, bit-identical output).
        The inner loop is NOT the Haxe-vs-Rust gap — do not "fix" it by copying
        the Rust shape. */
    static inline function int8DotRow(wBase:Usize, wOff:Int, aBase:Usize,
            aOff:Int, K:Int):Int {
        var acc0 = SIMD4i32.splat(0);
        var acc1 = SIMD4i32.splat(0);
        var acc2 = SIMD4i32.splat(0);
        var acc3 = SIMD4i32.splat(0);
        // POINTER-WALK. Advancing two i64 base pointers once per iteration
        // gives the backend clean induction variables to post-increment, and
        // the +16/+32/+48 are CONSTANT i64 offsets it folds into the load
        // ([ptr, #16]). The prior `base + Usize.fromInt(aOff + k + C)` form put
        // the constant INSIDE a 32-bit index add — so the offset could not be
        // hoisted past the sign-extend, and every load recomputed its address
        // (add + sxtw, ~10 adds/iter by disasm) vs Rust's pointer-walk.
        var aP = aBase + Usize.fromInt(aOff);
        var wP = wBase + Usize.fromInt(wOff);
        var o16 = Usize.fromInt(16);
        var o32 = Usize.fromInt(32);
        var o48 = Usize.fromInt(48);
        var o64 = Usize.fromInt(64);
        var k = 0;
        while (k + 64 <= K) {
            acc0 = SIMD4i32.dot(acc0,
                SIMD16i8.load(Ptr.fromRaw(aP)),
                SIMD16i8.load(Ptr.fromRaw(wP)));
            acc1 = SIMD4i32.dot(acc1,
                SIMD16i8.load(Ptr.fromRaw(aP + o16)),
                SIMD16i8.load(Ptr.fromRaw(wP + o16)));
            acc2 = SIMD4i32.dot(acc2,
                SIMD16i8.load(Ptr.fromRaw(aP + o32)),
                SIMD16i8.load(Ptr.fromRaw(wP + o32)));
            acc3 = SIMD4i32.dot(acc3,
                SIMD16i8.load(Ptr.fromRaw(aP + o48)),
                SIMD16i8.load(Ptr.fromRaw(wP + o48)));
            aP = aP + o64;
            wP = wP + o64;
            k += 64;
        }
        // Remaining 16-lane chunks fold into acc0.
        while (k + 16 <= K) {
            acc0 = SIMD4i32.dot(acc0,
                SIMD16i8.load(Ptr.fromRaw(aP)),
                SIMD16i8.load(Ptr.fromRaw(wP)));
            aP = aP + o16;
            wP = wP + o16;
            k += 16;
        }
        var s = acc0.sum() + acc1.sum() + acc2.sum() + acc3.sum();
        var o1 = Usize.fromInt(1);
        while (k < K) {
            var wv = Mem.loadU8(wP);
            if (wv > 127) wv -= 256;
            var av = Mem.loadU8(aP);
            if (av > 127) av -= 256;
            s += wv * av;
            aP = aP + o1;
            wP = wP + o1;
            k++;
        }
        return s;
    }

    /** `y = x @ qw.T` for an INT8 (per-row) weight, pure Haxe. Replaces the
        matmulXTQThreaded FFI bail. */
    static function matmulInt8(qw:QTensor, x:Tensor, sp:Null<SpinPool>):Tensor {
        var K = qw.cols();
        var batch = Std.int(x.numel() / K);
        if (batch < 1) batch = 1;

        var pooled = sp != null;
        var qs = pooled ? sp.scratchBytes(0, batch * K) : Bytes.alloc(batch * K);
        var aBase = qs.address();
        var xScales = pooled ? sp.scratchBytes(2, batch * 4) : Bytes.alloc(batch * 4);
        var sBase = xScales.address();
        // Time the shared activation quantise like the k-quant path does —
        // without this the row-wise kernels report quant_ms=0 and the split
        // between quantise and band is invisible.
        var _prof = sp != null && sp.profiling();
        var _tq = _prof ? Sys.time() : 0.0;
        quantiseRowsI8(x.data().raw(), aBase, sBase, batch, K);
        if (_prof) sp.addQuantUs(Std.int((Sys.time() - _tq) * 1e6));

        var y = int8Banded(qw, batch, K, aBase, sBase, sp);
        if (!pooled) {
            qs.free();
            xScales.free();
        }
        return y;
    }

    /** Band an INT8 weight against ALREADY-quantised activations. Row scale
        folds in once per output element: `y[b,n] = xScale[b]*wScale[n]*dot`. */
    /** ONE Q8_0 output row: 4 blocks folded through a SIMD4f, scalar tail.
        Hoisted out of `q8Banded`'s local closure so the fused segment path
        (`q8Seg`) uses the SAME definition — two copies of this math would be
        two places for the fused and unfused results to drift apart. */
    static inline function q8RowDot(wBase:Usize, rowBase:Int, aBase:Usize,
            aRow:Int, blocks:Int, blockBytes:Int, blockElems:Int):Float {
        var acc = 0.0;
        var blk = 0;
        var vacc = SIMD4f.splat(0.0);
        while (blk + 4 <= blocks) {
            var o0 = rowBase + blk * blockBytes;
            var o1 = o0 + blockBytes;
            var o2 = o1 + blockBytes;
            var o3 = o2 + blockBytes;
            var e0 = aRow + blk * blockElems;
            var vd = SIMD4f.make(
                f16ToF32(Mem.loadI32(wBase + Usize.fromInt(o0)) & 0xFFFF),
                f16ToF32(Mem.loadI32(wBase + Usize.fromInt(o1)) & 0xFFFF),
                f16ToF32(Mem.loadI32(wBase + Usize.fromInt(o2)) & 0xFFFF),
                f16ToF32(Mem.loadI32(wBase + Usize.fromInt(o3)) & 0xFFFF));
            var vs = SIMD4f.make(
                q8DotBlock(wBase, o0, aBase, e0),
                q8DotBlock(wBase, o1, aBase, e0 + blockElems),
                q8DotBlock(wBase, o2, aBase, e0 + 2 * blockElems),
                q8DotBlock(wBase, o3, aBase, e0 + 3 * blockElems));
            vacc = vacc.add(vd.mul(vs));
            blk += 4;
        }
        acc = vacc.get(0) + vacc.get(1) + vacc.get(2) + vacc.get(3);
        while (blk < blocks) {
            var o0 = rowBase + blk * blockBytes;
            var d0 = f16ToF32(Mem.loadI32(wBase + Usize.fromInt(o0)) & 0xFFFF);
            acc += d0 * q8DotBlock(wBase, o0, aBase, aRow + blk * blockElems);
            blk++;
        }
        return acc;
    }

    /** Q8_0 counterpart of `int8Seg`: this weight's clipped slice of a fused
        row space, same 4-row unroll as `q8Banded`. */
    static function q8Seg(n0:Int, n1:Int, segStart:Int, segEnd:Int, wBase:Usize,
            yBase:Usize, rows:Int, batch:Int, K:Int, aBase:Usize, sBase:Usize,
            blocks:Int, blockBytes:Int, blockElems:Int):Void {
        var lo = (n0 > segStart) ? n0 : segStart;
        var hi = (n1 < segEnd) ? n1 : segEnd;
        if (lo >= hi) return;
        var rowBytes = blocks * blockBytes;
        for (b in 0...batch) {
            var aRow = b * K;
            var xs = Mem.loadF32(sBase + Usize.fromInt(b << 2));
            var n = lo;
            while (n + 4 <= hi) {
                var l = n - segStart;
                var s0 = q8RowDot(wBase, l * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                var s1 = q8RowDot(wBase, (l + 1) * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                var s2 = q8RowDot(wBase, (l + 2) * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                var s3 = q8RowDot(wBase, (l + 3) * rowBytes, aBase, aRow, blocks, blockBytes, blockElems);
                var o = (b * rows + l) << 2;
                Mem.storeF32(yBase + Usize.fromInt(o), s0 * xs);
                Mem.storeF32(yBase + Usize.fromInt(o + 4), s1 * xs);
                Mem.storeF32(yBase + Usize.fromInt(o + 8), s2 * xs);
                Mem.storeF32(yBase + Usize.fromInt(o + 12), s3 * xs);
                n += 4;
            }
            while (n < hi) {
                var l = n - segStart;
                Mem.storeF32(yBase + Usize.fromInt((b * rows + l) << 2),
                    q8RowDot(wBase, l * rowBytes, aBase, aRow, blocks, blockBytes, blockElems) * xs);
                n++;
            }
        }
    }

    /** One weight's slice of a fused row space: clip the dispatched band
        `[n0,n1)` to this weight's `[segStart,segEnd)` and run the SAME 4-row
        unrolled loop `int8Banded` uses, so the math and reduction order (and
        therefore the output bits) are unchanged. Local row = global - segStart.

        Segment-clipping rather than `runBandedFused`'s per-row weight lookup:
        that form branches on every row and cannot keep the 4-row unroll, which
        is what feeds the scheduler 16 independent SDOT chains. */
    static function int8Seg(n0:Int, n1:Int, segStart:Int, segEnd:Int,
            wBase:Usize, wScaleBase:Usize, yBase:Usize, rows:Int, batch:Int,
            kk:Int, aBase:Usize, sBase:Usize):Void {
        var lo = (n0 > segStart) ? n0 : segStart;
        var hi = (n1 < segEnd) ? n1 : segEnd;
        if (lo >= hi) return;
        for (b in 0...batch) {
            var aRow = b * kk;
            var xs = Mem.loadF32(sBase + Usize.fromInt(b << 2));
            var n = lo;
            while (n + 4 <= hi) {
                var l = n - segStart;
                var d0 = int8DotRow(wBase, l * kk, aBase, aRow, kk);
                var d1 = int8DotRow(wBase, (l + 1) * kk, aBase, aRow, kk);
                var d2 = int8DotRow(wBase, (l + 2) * kk, aBase, aRow, kk);
                var d3 = int8DotRow(wBase, (l + 3) * kk, aBase, aRow, kk);
                var o = (b * rows + l) << 2;
                Mem.storeF32(yBase + Usize.fromInt(o),
                    xs * Mem.loadF32(wScaleBase + Usize.fromInt(l << 2)) * d0);
                Mem.storeF32(yBase + Usize.fromInt(o + 4),
                    xs * Mem.loadF32(wScaleBase + Usize.fromInt((l + 1) << 2)) * d1);
                Mem.storeF32(yBase + Usize.fromInt(o + 8),
                    xs * Mem.loadF32(wScaleBase + Usize.fromInt((l + 2) << 2)) * d2);
                Mem.storeF32(yBase + Usize.fromInt(o + 12),
                    xs * Mem.loadF32(wScaleBase + Usize.fromInt((l + 3) << 2)) * d3);
                n += 4;
            }
            while (n < hi) {
                var l = n - segStart;
                var ws = Mem.loadF32(wScaleBase + Usize.fromInt(l << 2));
                Mem.storeF32(yBase + Usize.fromInt((b * rows + l) << 2),
                    xs * ws * int8DotRow(wBase, l * kk, aBase, aRow, kk));
                n++;
            }
        }
    }

    /** Band an all-INT8 triple in ONE pool dispatch over the CONCATENATED row
        space, instead of one dispatch per weight.

        The measured cost is fork/join, not math: a 300-token request issued
        55366 dispatches (~184/token, ~7.7 per layer — QKV 3 + O 1 + gate/up 2 +
        down 1). Collapsing QKV to one and gate/up to one takes that to ~4 per
        layer. Sharing the activation quantise (which this path already did) was
        worth almost nothing by comparison — it measures 0.1% of band time. */
    static function rowwiseBandedFused(w0:QTensor, w1:QTensor, w2:QTensor,
            batch:Int, K:Int, aBase:Usize, sBase:Usize,
            sp:Null<SpinPool>):Array<Tensor> {
        var r0 = w0.rows();
        var r1 = w1.rows();
        var r2 = (w2 != null) ? w2.rows() : 0;
        var e0 = r0;
        var e1 = r0 + r1;
        var total = r0 + r1 + r2;

        // Raw addresses only — the band closure must never capture the
        // refcounted QTensors (the capture-set rule; ARC traffic across a hot
        // banded pass cost the Q8_0 kernel 2.5x).
        // Per-weight scheme: INT8 and Q8_0 share the activation encoding but
        // not the weight layout, so each segment picks its own row kernel.
        // scalesPtr is INT8-only (Q8_0 carries a per-block f16 scale inline).
        var i0 = (w0.scheme() == QScheme.INT8);
        var i1 = (w1.scheme() == QScheme.INT8);
        var i2 = (w2 != null) && (w2.scheme() == QScheme.INT8);
        var zero = Usize.fromInt(0);
        var blocks = Std.int(K / Q8_0_BLOCK_ELEMS);
        var bbytes = Q8_0_BLOCK_BYTES;
        var belems = Q8_0_BLOCK_ELEMS;

        var wb0 = w0.dataPtr();
        var sc0 = i0 ? w0.scalesPtr() : zero;
        var wb1 = w1.dataPtr();
        var sc1 = i1 ? w1.scalesPtr() : zero;
        var wb2 = (w2 != null) ? w2.dataPtr() : wb0;
        var sc2 = (w2 != null && w2.scheme() == QScheme.INT8) ? w2.scalesPtr() : sc0;

        var y0 = Tensor.uninit([batch, r0], DType.F32);
        var y1 = Tensor.uninit([batch, r1], DType.F32);
        var y2 = (w2 != null) ? Tensor.uninit([batch, r2], DType.F32) : null;
        var yb0 = y0.data().raw();
        var yb1 = y1.data().raw();
        var yb2 = (w2 != null) ? y2.data().raw() : yb0;
        var kk = K;
        var bt = batch;
        var hasW2 = (w2 != null);

        var band = function(n0:Int, n1:Int, node:Int):Void {
            if (i0) int8Seg(n0, n1, 0, e0, wb0, sc0, yb0, r0, bt, kk, aBase, sBase);
            else q8Seg(n0, n1, 0, e0, wb0, yb0, r0, bt, kk, aBase, sBase, blocks, bbytes, belems);
            if (i1) int8Seg(n0, n1, e0, e1, wb1, sc1, yb1, r1, bt, kk, aBase, sBase);
            else q8Seg(n0, n1, e0, e1, wb1, yb1, r1, bt, kk, aBase, sBase, blocks, bbytes, belems);
            if (hasW2) {
                if (i2) int8Seg(n0, n1, e1, total, wb2, sc2, yb2, r2, bt, kk, aBase, sBase);
                else q8Seg(n0, n1, e1, total, wb2, yb2, r2, bt, kk, aBase, sBase, blocks, bbytes, belems);
            }
        };
        if (sp != null) sp.parallelRows(total, band);
        else band(0, total, 0);

        var out = [y0, y1];
        if (y2 != null) out.push(y2);
        return out;
    }

    static function int8Banded(qw:QTensor, batch:Int, K:Int, aBase:Usize,
            sBase:Usize, sp:Null<SpinPool>):Tensor {
        var rows = qw.rows();
        // Hoist to RAW addresses / plain Ints: the band closure must capture
        // only Usize/Int, never the refcounted `qw` (ARC traffic across a hot
        // banded pass cost the Q8_0 kernel 2.5x — the capture-set rule).
        var wBase = qw.dataPtr();
        var wScaleBase = qw.scalesPtr();
        var kk = K;
        var y = Tensor.uninit([batch, rows], DType.F32);
        var yBase = y.data().raw();

        var band = function(n0:Int, n1:Int, node:Int):Void {
            for (b in 0...batch) {
                var aRow = b * kk;
                var xs = Mem.loadF32(sBase + Usize.fromInt(b << 2));
                var n = n0;
                // FOUR output rows per step: int8DotRow is inline, so four calls
                // give the scheduler four independent row dots (16 SDOT chains
                // total with the per-row 4-accumulator split) to overlap, and
                // the shared activation row stays L1-hot across all four — the
                // same shape that brought the Q8_0 band to parity.
                while (n + 4 <= n1) {
                    var d0 = int8DotRow(wBase, n * kk, aBase, aRow, kk);
                    var d1 = int8DotRow(wBase, (n + 1) * kk, aBase, aRow, kk);
                    var d2 = int8DotRow(wBase, (n + 2) * kk, aBase, aRow, kk);
                    var d3 = int8DotRow(wBase, (n + 3) * kk, aBase, aRow, kk);
                    var o = (b * rows + n) << 2;
                    Mem.storeF32(yBase + Usize.fromInt(o),
                        xs * Mem.loadF32(wScaleBase + Usize.fromInt(n << 2)) * d0);
                    Mem.storeF32(yBase + Usize.fromInt(o + 4),
                        xs * Mem.loadF32(wScaleBase + Usize.fromInt((n + 1) << 2)) * d1);
                    Mem.storeF32(yBase + Usize.fromInt(o + 8),
                        xs * Mem.loadF32(wScaleBase + Usize.fromInt((n + 2) << 2)) * d2);
                    Mem.storeF32(yBase + Usize.fromInt(o + 12),
                        xs * Mem.loadF32(wScaleBase + Usize.fromInt((n + 3) << 2)) * d3);
                    n += 4;
                }
                while (n < n1) {
                    var ws = Mem.loadF32(wScaleBase + Usize.fromInt(n << 2));
                    Mem.storeF32(yBase + Usize.fromInt((b * rows + n) << 2),
                        xs * ws * int8DotRow(wBase, n * kk, aBase, aRow, kk));
                    n++;
                }
            }
        };
        if (sp != null) sp.parallelRows(rows, band);
        else band(0, rows, 0);
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
        // Hoist to a RAW ADDRESS so the band closure below captures only
        // Usize/Int — never the refcounted `bsums` Bytes. A hot closure that
        // captures a refcounted object pays ARC traffic across the whole banded
        // pass: the identical shape cost the Q8_0 kernel 2.5x (36.7 -> ~90).
        var bsAddr = bsums.address();
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
                            sum += q6DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsAddr, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)))
                                 + q6DotMA4(wBase, (base + b + 1) * blockBytes, aBase, (b + 1) * 256, bsAddr, (b + 1) * 16, Mem.loadF32(dBase + Usize.fromInt((b + 1) << 2)));
                            b += 2;
                        }
                        while (b < bpr) {
                            sum += q6DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsAddr, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)));
                            b++;
                        }
                    } else {
                        while (b + 2 <= bpr) {
                            sum += q4DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsAddr, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)))
                                 + q4DotMA4(wBase, (base + b + 1) * blockBytes, aBase, (b + 1) * 256, bsAddr, (b + 1) * 16, Mem.loadF32(dBase + Usize.fromInt((b + 1) << 2)));
                            b += 2;
                        }
                        while (b < bpr) {
                            sum += q4DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsAddr, b * 16, Mem.loadF32(dBase + Usize.fromInt(b << 2)));
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
                            s0 += q6DotMA4(wBase, blkOff, aBase, a0, bsAddr, i0, d0);
                            s1 += q6DotMA4(wBase, blkOff, aBase, a0 + K, bsAddr, i0 + bpr * 16, d1);
                            s2 += q6DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsAddr, i0 + 2 * bpr * 16, d2);
                            s3 += q6DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsAddr, i0 + 3 * bpr * 16, d3);
                        } else {
                            s0 += q4DotMA4(wBase, blkOff, aBase, a0, bsAddr, i0, d0);
                            s1 += q4DotMA4(wBase, blkOff, aBase, a0 + K, bsAddr, i0 + bpr * 16, d1);
                            s2 += q4DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsAddr, i0 + 2 * bpr * 16, d2);
                            s3 += q4DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsAddr, i0 + 3 * bpr * 16, d3);
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
                            ? q6DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsAddr, (rt * bpr + b) * 16, xdb)
                            : q4DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsAddr, (rt * bpr + b) * 16, xdb);
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
        // Any INT8 per-row weight (Q5_0-sourced) drops the triple to
        // per-weight calls — `matmul` routes each INT8 weight to the
        // runtime XTQ path, and the banded/fused machinery below is
        // k-quant-only.
        // Row-wise/32-block schemes (INT8 from Q5_0/Q5_1, and native Q8_0)
        // have no place in the 256-wide banded fused path below.
        var anyRowwise = isRowwise(w0) || isRowwise(w1) || (w2 != null && isRowwise(w2));

        // ROW-WISE triples (INT8 and/or native Q8_0) get their OWN fused path.
        // Q/K/V share the post-attn-norm activation and gate/up share the
        // post-FFN-norm one, so the split fallback re-runs the same two-pass
        // O(K) int8 quantise PER WEIGHT — ~5 quantisations per layer where 2
        // would do. Quantise once, band each weight against the shared buffer.
        //
        // INT8 and Q8_0 are fused TOGETHER because they share one activation
        // encoding: both consume per-row int8 from `quantiseRowsI8`, and only
        // the WEIGHT side differs (per-row scale vs per-32-block f16 scale), so
        // each weight just picks its own band kernel. It is the K-QUANT schemes
        // that cannot join — those need Q8_K super-blocks (`quantizeAll`), a
        // different buffer entirely — so a mixed rowwise+k-quant triple still
        // falls through to the split path below.
        if (canFuseRowwise(w0, w1, w2)) {
            var K = w0.cols();
            var batch = Std.int(x.numel() / K);
            if (batch < 1) batch = 1;
            var pooled = sp != null;
            var qs = pooled ? sp.scratchBytes(0, batch * K) : Bytes.alloc(batch * K);
            var aBase = qs.address();
            var xScales = pooled ? sp.scratchBytes(2, batch * 4) : Bytes.alloc(batch * 4);
            var sBase = xScales.address();
            // Time the shared quantise like the k-quant path does — without it
            // the row-wise kernels report quant_ms=0 and the quantise/band
            // split is invisible. Distinct names: this function declares its
            // own `_prof` further down for the k-quant branch.
            var _profR = sp != null && sp.profiling();
            var _tqR = _profR ? Sys.time() : 0.0;
            quantiseRowsI8(x.data().raw(), aBase, sBase, batch, K);
            if (_profR) sp.addQuantUs(Std.int((Sys.time() - _tqR) * 1e6));
            // All-INT8: band the CONCATENATED row space in ONE dispatch. Mixed
            // INT8/Q8_0 triples keep the per-weight path — their band kernels
            // differ, so they cannot share a single row space.
            var out:Array<Tensor>;
            planFused(w2 != null, useFusedDispatch());
            if (useFusedDispatch()) {
                // Any row-wise triple (INT8 and/or Q8_0) — each segment picks
                // its own row kernel, so mixed triples fuse too.
                census(w0.scheme() == QScheme.INT8 ? 0 : 3, false);
                census(w1.scheme() == QScheme.INT8 ? 0 : 3, false);
                if (w2 != null) census(w2.scheme() == QScheme.INT8 ? 0 : 3, false);
                out = rowwiseBandedFused(w0, w1, w2, batch, K, aBase, sBase, sp);
            } else {
                out = [rowwiseBand(w0, batch, K, aBase, sBase, sp),
                       rowwiseBand(w1, batch, K, aBase, sBase, sp)];
                if (w2 != null) out.push(rowwiseBand(w2, batch, K, aBase, sBase, sp));
            }
            if (!pooled) {
                qs.free();
                xScales.free();
            }
            return out;
        }

        if (!useFusedMatmul() || anyRowwise) {
            planSplit(w2 != null);
            var split = [matmul(w0, x, sp), matmul(w1, x, sp)];
            if (w2 != null) split.push(matmul(w2, x, sp));
            return split;
        }
        var K = w0.cols();
        var batch = Std.int(x.numel() / K);
        if (batch != 1) {
            // Opt-in AMX prefill experiment (see matmul): route each fused
            // weight through the runtime kernel. Q4_K_M only — a Q6_K weight
            // (attn_v in the QKV triple) would hit the runtime's legacy
            // fallback; keep such triples on the banded path. matmulXTQThreaded
            // does NOT free x, so the shared activation survives all calls.
            var allQ4 = (w0.scheme() == QScheme.Q4_K_M) && (w1.scheme() == QScheme.Q4_K_M)
                && (w2 == null || w2.scheme() == QScheme.Q4_K_M);
            if (batch >= amxMinBatch() && amxPrefill() && allQ4) {
                var outs = [w0.matmulXTQThreaded(x, 0), w1.matmulXTQThreaded(x, 0)];
                if (w2 != null) outs.push(w2.matmulXTQThreaded(x, 0));
                return outs;
            }
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
        var bsAddr = bsums.address(); // raw addr for MA4 calls (no Bytes in hot scope)
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
                        ? q6DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsAddr, b * 16, xdb)
                        : q4DotMA4(wBase, blk * blockBytes, aBase, b * 256, bsAddr, b * 16, xdb);
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
        var bsAddr = bsums.address(); // raw addr: band closures capture Usize only
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
                            ? q6DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsAddr, b * 16, xdb)
                            : q4DotMA4(wBase, (base + b) * blockBytes, aBase, b * 256, bsAddr, b * 16, xdb);
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
                            s0 += q6DotMA4(wBase, blkOff, aBase, a0, bsAddr, i0, d0);
                            s1 += q6DotMA4(wBase, blkOff, aBase, a0 + K, bsAddr, i0 + bpr * 16, d1);
                            s2 += q6DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsAddr, i0 + 2 * bpr * 16, d2);
                            s3 += q6DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsAddr, i0 + 3 * bpr * 16, d3);
                        } else {
                            s0 += q4DotMA4(wBase, blkOff, aBase, a0, bsAddr, i0, d0);
                            s1 += q4DotMA4(wBase, blkOff, aBase, a0 + K, bsAddr, i0 + bpr * 16, d1);
                            s2 += q4DotMA4(wBase, blkOff, aBase, a0 + 2 * K, bsAddr, i0 + 2 * bpr * 16, d2);
                            s3 += q4DotMA4(wBase, blkOff, aBase, a0 + 3 * K, bsAddr, i0 + 3 * bpr * 16, d3);
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
                            ? q6DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsAddr, (rt * bpr + b) * 16, xdb)
                            : q4DotMA4(wBase, blkOff, aBase, rt * K + b * 256, bsAddr, (rt * bpr + b) * 16, xdb);
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
