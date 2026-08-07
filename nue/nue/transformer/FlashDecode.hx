package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Mem;
import rayzor.Bytes;
import rayzor.concurrent.SpinPool;

/**
 * Decode-time flash attention over the pure-Haxe Q8_0 KV cache ([Q8Cache]).
 *
 * One query row per q-head attends over `cacheLen` cached positions.
 * Work bands across kv-heads: a band owns its whole GQA group's score
 * and output rows outright, so bands share nothing but read-only cache
 * bytes.
 */
class FlashDecode {
    // 0 = unread, 1 = on, 2 = off. Int (not a -1 sentinel): a cross-module
    // duplicate of this static starts at 0 and re-reads the env instead of
    // wrongly latching "off".
    static var _enabled:Int = 0;
    static var _shiftedQ:Int = 0;
    static var _pool:Int = 0;

    /** The pure-Haxe decode-attention path. DEFAULT ON; `NUE_FLASH=0` opts out.

        Must agree with `LlamaArch`'s reading of the same variable — that one
        decides whether the cache is BUILT as guest-owned Q8, this one decides
        whether the kernel is USED. Two gates on one variable with opposite
        defaults would build a Q8 cache nothing reads. */
    public static function enabled():Bool {
        if (_enabled == 0) {
            var v = Sys.getEnvOr("NUE_FLASH", "RAYZOR_HAXE_FLASH");
            _enabled = (v != null && (v == "0" || v == "false")) ? 2 : 1;
        }
        return _enabled == 1;
    }

    static function shiftedQ():Bool {
        if (_shiftedQ == 0) {
            var v = Sys.getEnvOr("NUE_FLASH_SHIFTED_Q", "RAYZOR_HAXE_FLASH_SHIFTED_Q");
            if (v != null && v != "") {
                _shiftedQ = (v != "0" && v.toLowerCase() != "false") ? 1 : 2;
            } else {
                // Keep the production path on signed Q8xQ8 attention. The
                // shifted-query VNNI form is useful for x86 experiments, but
                // it is too risky as a default because a model-quality miss
                // here corrupts every decode step.
                _shiftedQ = 2;
            }
        }
        return _shiftedQ == 1;
    }

    static function usePool():Bool {
        if (_pool == 0) {
            var v = Sys.getEnvOr("NUE_FLASH_POOL", "RAYZOR_HAXE_FLASH_POOL");
            if (v == null || v == "") {
                // Keep decode flash parallel by default. Long-prompt llama
                // runs on Apple Silicon lose a measurable slice of throughput
                // when these kv-head bands run serially; callers can still
                // set RAYZOR_HAXE_FLASH_POOL=0 for A/B or tiny-core devices.
                _pool = 1;
            } else {
                _pool = (v != "0" && v.toLowerCase() != "false") ? 1 : 2;
            }
        }
        return _pool == 1;
    }

    /** Batch flash is explicitly disabled on the default path. The prior
        seqQ>1/shifted-query experiment regressed full-model correctness; keep
        single-token flash live and make any future batch work opt in from a
        fresh, isolated implementation. */
    /** Longest prompt the batched (prefill) path will take. 0 = disabled.
        Prefill attention otherwise leaves Haxe entirely: it falls through to
        `Tensor.bmmThreaded`, which runs on the RUST worker pool — measured at
        23.5% of prefill, with a further ~18% lost to that pool and the Haxe
        SpinPool spinning at each other across the handoff. */
    // 0 = unread; stored as value+1 so a CROSS-MODULE DUPLICATE of this static
    // (which starts at 0) re-reads the env instead of latching "disabled" —
    // the same reason `_enabled`/`_pool` above avoid a -1 sentinel.
    static var _batchMax:Int = 0;

    public static function batchMax():Int {
        if (_batchMax == 0) {
            var v = Sys.getEnvOr("NUE_FLASH_BATCH", "RAYZOR_HAXE_FLASH_BATCH");
            var n = (v != null && v != "") ? Std.int(Std.parseFloat(v)) : 0;
            if (n < 0) n = 0;
            _batchMax = n + 1;
        }
        return _batchMax - 1;
    }

    // Batched-prefill scratch. Grown on demand and reused; sized per WORKER
    // for the score/V buffers because rows are banded across the pool.
    static var _bq:Bytes = null;
    static var _bqs:Bytes = null;
    static var _bsc:Bytes = null;
    static var _bv:Bytes = null;

    static inline function grow(b:Bytes, need:Int):Bytes {
        if (b != null && b.length >= need) return b;
        if (b != null) b.free();
        return Bytes.alloc(need);
    }

    /**
     * Batched (prefill) Q8 attention: `q` is [seqQ, numQHeads, headDim] F32
     * contiguous, and the K/V for these rows are ALREADY in the cache, so
     * query row `r` attends over keys `[0, baseLen + r]` — the causal mask is
     * just a per-row cache length, no mask tensor.
     *
     * Returns a fresh [seqQ, numQHeads, headDim] F32 tensor, or null to fall
     * back (shifted-Q form and the no-pool case are not implemented here).
     */
    public static function decodeBatch(kc:Q8Cache, vc:Q8Cache, q:Tensor,
            baseLen:Int, seqQ:Int, numQHeads:Int, scale:Float,
            sp:Null<SpinPool>):Tensor {
        if (sp == null || !usePool() || shiftedQ() || seqQ < 1) return null;
        var numKvHeads = kc.numKvHeads;
        var headDim = kc.headDim;
        if (numQHeads % numKvHeads != 0) return null;
        var group = Std.int(numQHeads / numKvHeads);
        var bph = headDim >> 5;
        if (bph < 1) return null;
        var rowStride = kc.maxSeqLen;
        if (baseLen + seqQ > rowStride) return null;

        var nw = sp.workers();
        if (nw < 1) nw = 1;
        _bq = grow(_bq, seqQ * numQHeads * headDim);
        _bqs = grow(_bqs, seqQ * numQHeads * bph * 4);
        _bsc = grow(_bsc, nw * numQHeads * rowStride * 4);
        _bv = grow(_bv, nw * numKvHeads * 128);

        var qBase:Usize = q.data().raw();
        var qqB:Usize = _bq.address();
        var qscB:Usize = _bqs.address();

        // Quantise every query row to per-32-block q8, exactly as `decode`
        // does for its single row.
        for (r in 0...seqQ) {
            for (qh in 0...numQHeads) {
                for (b in 0...bph) {
                    var src = qBase + Usize.fromInt(((r * numQHeads + qh) * headDim + b * 32) * 4);
                    var maxAbs = 0.0;
                    for (i in 0...32) {
                        var v = Mem.loadF32(src + Usize.fromInt(i * 4));
                        var nv = -v;
                        var av = v > nv ? v : nv;
                        maxAbs = av > maxAbs ? av : maxAbs;
                    }
                    // Identical numerics to `decode`'s quantiser — same scale,
                    // same rounding, same clamp, same unsigned byte store.
                    var qs = maxAbs == 0.0 ? 0.0 : maxAbs / 127.0;
                    var inv = qs == 0.0 ? 0.0 : 1.0 / qs;
                    Mem.storeF32(qscB + Usize.fromInt(((r * numQHeads + qh) * bph + b) * 4), qs);
                    var dst = qqB + Usize.fromInt((r * numQHeads + qh) * headDim + b * 32);
                    for (i in 0...32) {
                        var x = Mem.loadF32(src + Usize.fromInt(i * 4)) * inv;
                        var iv = x >= 0 ? Std.int(x + 0.5) : Std.int(x - 0.5);
                        if (iv > 127) iv = 127;
                        if (iv < -128) iv = -128;
                        Mem.storeU8(dst + Usize.fromInt(i), iv & 0xFF);
                    }
                }
            }
        }

        var out = Tensor.zeros([seqQ, numQHeads, headDim], DType.F32);
        var outB:Usize = out.data().raw();
        var kBase:Usize = kc.data.address();
        var vBase:Usize = vc.data.address();
        var kScale:Usize = kc.scaleF32.address();
        var vScale:Usize = vc.scaleF32.address();
        var headBytes = kc.headBytes;
        var rowBytes = kc.rowBytes;
        var blocksPerRow:Int = kc.blocksPerRow;
        var scBase:Usize = _bsc.address();
        var vBaseScr:Usize = _bv.address();
        var qRowBytes = numQHeads * headDim;
        var qscRowFloats = numQHeads * bph;
        var outRowFloats = numQHeads * headDim;

        // Band over query ROWS: each row is independent and owns its output
        // row, so the only shared state is the per-worker score/V scratch.
        var band = function(lo:Int, hi:Int, w:Int):Void {
            var scW = scBase + Usize.fromInt(w * numQHeads * rowStride * 4);
            var vW = vBaseScr + Usize.fromInt(w * numKvHeads * 128);
            for (r in lo...hi) {
                var cacheLen = baseLen + r + 1;
                var qqR = qqB + Usize.fromInt(r * qRowBytes);
                var qscR = qscB + Usize.fromInt(r * qscRowFloats * 4);
                var outR = outB + Usize.fromInt(r * outRowFloats * 4);
                for (h in 0...numKvHeads) {
                    bandOneSigned(h, group, bph, headBytes, rowBytes, cacheLen,
                        rowStride, headDim, scale, kBase, vBase, qqR, qscR, scW,
                        vW, outR, kScale, vScale, blocksPerRow);
                }
            }
        };
        sp.parallelRows(seqQ, band);
        if (Sys.getEnvOr("NUE_FLASH_BATCH_DEBUG", "RAYZOR_FLASH_BATCH_DEBUG") != null) {
            Sys.println("[flash-batch] seqQ=" + seqQ + " baseLen=" + baseLen
                + " qHeads=" + numQHeads + " kvHeads=" + numKvHeads + " workers=" + nw);
        }
        return out;
    }

    /**
     * q: [1, numQHeads, headDim] F32 contiguous. Returns a fresh
     * [1, numQHeads, headDim] F32 tensor the caller frees.
     */
    public static function decode(kc:Q8Cache, vc:Q8Cache, q:Tensor, cacheLen:Int,
            numQHeads:Int, scale:Float, sp:Null<SpinPool>):Tensor {
        var numKvHeads = kc.numKvHeads;
        var headDim = kc.headDim;
        var group = Std.int(numQHeads / numKvHeads);
        var bph = headDim >> 5;
        var headBytes = kc.headBytes;
        var rowBytes = kc.rowBytes;
        var kBase:Usize = kc.data.address();
        var vBase:Usize = vc.data.address();
        var qBase:Usize = q.data().raw();
        var kScale:Usize = kc.scaleF32.address();
        var vScale:Usize = vc.scaleF32.address();
        var kSum:Usize = kc.sumI32.address();
        var blocksPerRow:Int = kc.blocksPerRow;
        var shifted = shiftedQ();

        var out = Tensor.zeros([1, numQHeads, headDim], DType.F32);
        var outB:Usize = out.data().raw();

        kc.ensureDecodeScratch(numQHeads);
        var rowStride = kc.maxSeqLen;
        var qqB:Usize = kc.scrQ.address();
        var qscB:Usize = kc.scrQScale.address();
        var scB:Usize = kc.scrScores.address();
        var vScrB:Usize = kc.scrV.address();

        // Quantize the q rows to per-32-block q8.
        for (qh in 0...numQHeads) {
            for (b in 0...bph) {
                var src = qBase + Usize.fromInt((qh * headDim + b * 32) * 4);
                var maxAbs = 0.0;
                for (i in 0...32) {
                    var v = Mem.loadF32(src + Usize.fromInt(i * 4));
                    var nv = -v;
                    var av = v > nv ? v : nv;
                    maxAbs = av > maxAbs ? av : maxAbs;
                }
                var s = maxAbs == 0.0 ? 0.0 : maxAbs / 127.0;
                var inv = s == 0.0 ? 0.0 : 1.0 / s;
                Mem.storeF32(qscB + Usize.fromInt((qh * bph + b) * 4), s);
                var dst = qqB + Usize.fromInt(qh * headDim + b * 32);
                for (i in 0...32) {
                    var x = Mem.loadF32(src + Usize.fromInt(i * 4)) * inv;
                    var r = x >= 0 ? Std.int(x + 0.5) : Std.int(x - 0.5);
                    if (r > 127) r = 127;
                    if (r < -128) r = -128;
                    if (shifted) {
                        // Store q+128. Pass 1 uses unsigned*signed VNNI as
                        // dot(K, q+128), then subtracts 128*sum(K) to recover
                        // signed Q8xQ8 exactly.
                        Mem.storeU8(dst + Usize.fromInt(i), (r + 128) & 0xFF);
                    } else {
                        Mem.storeU8(dst + Usize.fromInt(i), r & 0xFF);
                    }
                }
            }
        }

        var band = function(lo:Int, hi:Int, w:Int):Void {
            for (h in lo...hi) {
                if (shifted) {
                    bandOneShifted(h, group, bph, headBytes, rowBytes, cacheLen,
                        rowStride, headDim, scale, kBase, vBase, qqB, qscB, scB,
                        vScrB, outB, kScale, vScale, kSum, blocksPerRow);
                } else {
                    bandOneSigned(h, group, bph, headBytes, rowBytes, cacheLen,
                        rowStride, headDim, scale, kBase, vBase, qqB, qscB, scB,
                        vScrB, outB, kScale, vScale, blocksPerRow);
                }
            }
        };
        if (sp != null && usePool()) {
            sp.parallelRows(numKvHeads, band);
        } else {
            band(0, numKvHeads, 0);
        }
        return out;
    }

    static function bandOneShifted(h:Int, group:Int, bph:Int, headBytes:Int,
            rowBytes:Int, cacheLen:Int, rowStride:Int, headDim:Int, scale:Float,
            kBase:Usize, vBase:Usize, qqB:Usize, qscB:Usize, scB:Usize,
            vScrB:Usize, outB:Usize, kScale:Usize, vScale:Usize,
            kSum:Usize, blocksPerRow:Int):Void {
        var g0 = h * group;
        var hb = h * bph;

        var l = 0;
        while (l < cacheLen) {
            var rowP = kBase + Usize.fromInt(l * rowBytes + h * headBytes);
            var scRow = kScale + Usize.fromInt((l * blocksPerRow + hb) * 4);
            var sumRow = kSum + Usize.fromInt((l * blocksPerRow + hb) * 4);
            for (b in 0...bph) {
                var blk = rowP + Usize.fromInt(b * 34);
                var ksc = Mem.loadF32(scRow + Usize.fromInt(b * 4));
                var ksum = Mem.loadI32(sumRow + Usize.fromInt(b * 4));
                var k0 = SIMD16i8.load(Ptr.fromRaw(blk + Usize.fromInt(2)));
                var k1 = SIMD16i8.load(Ptr.fromRaw(blk + Usize.fromInt(18)));
                for (gi in 0...group) {
                    var qh = g0 + gi;
                    var qOff = qqB + Usize.fromInt(qh * headDim + b * 32);
                    var acc = SIMD4i32.splat(0);
                    acc = SIMD4i32.dotI8U8(acc, k0, SIMD16i8.load(Ptr.fromRaw(qOff)));
                    acc = SIMD4i32.dotI8U8(acc, k1,
                        SIMD16i8.load(Ptr.fromRaw(qOff + Usize.fromInt(16))));
                    var qs = Mem.loadF32(qscB + Usize.fromInt((qh * bph + b) * 4));
                    var p = ksc * qs * scale * (acc.sum() - 128 * ksum);
                    var cellA = scB + Usize.fromInt((qh * rowStride + l) * 4);
                    if (b == 0) {
                        Mem.storeF32(cellA, p);
                    } else {
                        Mem.storeF32(cellA, Mem.loadF32(cellA) + p);
                    }
                }
            }
            l++;
        }

        for (gi in 0...group) {
            var qh = g0 + gi;
            var rowA = scB + Usize.fromInt(qh * rowStride * 4);
            var mx = Mem.loadF32(rowA);
            var i = 1;
            while (i < cacheLen) {
                var v = Mem.loadF32(rowA + Usize.fromInt(i * 4));
                if (v > mx) mx = v;
                i++;
            }
            var denom = 0.0;
            i = 0;
            while (i < cacheLen) {
                var e = Math.exp(Mem.loadF32(rowA + Usize.fromInt(i * 4)) - mx);
                Mem.storeF32(rowA + Usize.fromInt(i * 4), e);
                denom += e;
                i++;
            }
            var invD = denom > 0 ? 1.0 / denom : 0.0;
            i = 0;
            while (i < cacheLen) {
                var a = rowA + Usize.fromInt(i * 4);
                Mem.storeF32(a, Mem.loadF32(a) * invD);
                i++;
            }
        }

        var vs = vScrB + Usize.fromInt(h * 128);
        l = 0;
        while (l < cacheLen) {
            var rowP = vBase + Usize.fromInt(l * rowBytes + h * headBytes);
            var vScRow = vScale + Usize.fromInt((l * blocksPerRow + hb) * 4);
            for (b in 0...bph) {
                var blk = rowP + Usize.fromInt(b * 34);
                var vsc = Mem.loadF32(vScRow + Usize.fromInt(b * 4));
                for (i in 0...32) {
                    var qv = (Mem.loadU8(blk + Usize.fromInt(2 + i)) << 24) >> 24;
                    Mem.storeF32(vs + Usize.fromInt(i * 4), vsc * qv);
                }
                for (gi in 0...group) {
                    var qh = g0 + gi;
                    var wgt = Mem.loadF32(scB + Usize.fromInt((qh * rowStride + l) * 4));
                    var ob = outB + Usize.fromInt((qh * headDim + b * 32) * 4);
                    var wv = SIMD4f.splat(wgt);
                    for (t in 0...8) {
                        var po = Ptr.fromRaw(ob + Usize.fromInt(t * 16));
                        var ov = SIMD4f.load(po);
                        var vv = SIMD4f.load(Ptr.fromRaw(vs + Usize.fromInt(t * 16)));
                        ov = ov.add(vv.mul(wv));
                        ov.store(po);
                    }
                }
            }
            l++;
        }
    }

    static function bandOneSigned(h:Int, group:Int, bph:Int, headBytes:Int,
            rowBytes:Int, cacheLen:Int, rowStride:Int, headDim:Int, scale:Float,
            kBase:Usize, vBase:Usize, qqB:Usize, qscB:Usize, scB:Usize,
            vScrB:Usize, outB:Usize, kScale:Usize, vScale:Usize,
            blocksPerRow:Int):Void {
        var g0 = h * group;
        var hb = h * bph;

        var l = 0;
        while (l < cacheLen) {
            var rowP = kBase + Usize.fromInt(l * rowBytes + h * headBytes);
            var scRow = kScale + Usize.fromInt((l * blocksPerRow + hb) * 4);
            for (b in 0...bph) {
                var blk = rowP + Usize.fromInt(b * 34);
                var ksc = Mem.loadF32(scRow + Usize.fromInt(b * 4));
                var k0 = SIMD16i8.load(Ptr.fromRaw(blk + Usize.fromInt(2)));
                var k1 = SIMD16i8.load(Ptr.fromRaw(blk + Usize.fromInt(18)));
                for (gi in 0...group) {
                    var qh = g0 + gi;
                    var qOff = qqB + Usize.fromInt(qh * headDim + b * 32);
                    var acc = SIMD4i32.splat(0);
                    acc = SIMD4i32.dot(acc, SIMD16i8.load(Ptr.fromRaw(qOff)), k0);
                    acc = SIMD4i32.dot(acc,
                        SIMD16i8.load(Ptr.fromRaw(qOff + Usize.fromInt(16))), k1);
                    var qs = Mem.loadF32(qscB + Usize.fromInt((qh * bph + b) * 4));
                    var p = ksc * qs * scale * acc.sum();
                    var cellA = scB + Usize.fromInt((qh * rowStride + l) * 4);
                    if (b == 0) {
                        Mem.storeF32(cellA, p);
                    } else {
                        Mem.storeF32(cellA, Mem.loadF32(cellA) + p);
                    }
                }
            }
            l++;
        }

        for (gi in 0...group) {
            var qh = g0 + gi;
            var rowA = scB + Usize.fromInt(qh * rowStride * 4);
            var mx = Mem.loadF32(rowA);
            var i = 1;
            while (i < cacheLen) {
                var v = Mem.loadF32(rowA + Usize.fromInt(i * 4));
                if (v > mx) mx = v;
                i++;
            }
            var denom = 0.0;
            i = 0;
            while (i < cacheLen) {
                var e = Math.exp(Mem.loadF32(rowA + Usize.fromInt(i * 4)) - mx);
                Mem.storeF32(rowA + Usize.fromInt(i * 4), e);
                denom += e;
                i++;
            }
            var invD = denom > 0 ? 1.0 / denom : 0.0;
            i = 0;
            while (i < cacheLen) {
                var a = rowA + Usize.fromInt(i * 4);
                Mem.storeF32(a, Mem.loadF32(a) * invD);
                i++;
            }
        }

        var vs = vScrB + Usize.fromInt(h * 128);
        l = 0;
        while (l < cacheLen) {
            var rowP = vBase + Usize.fromInt(l * rowBytes + h * headBytes);
            var vScRow = vScale + Usize.fromInt((l * blocksPerRow + hb) * 4);
            for (b in 0...bph) {
                var blk = rowP + Usize.fromInt(b * 34);
                var vsc = Mem.loadF32(vScRow + Usize.fromInt(b * 4));
                for (i in 0...32) {
                    var qv = (Mem.loadU8(blk + Usize.fromInt(2 + i)) << 24) >> 24;
                    Mem.storeF32(vs + Usize.fromInt(i * 4), vsc * qv);
                }
                for (gi in 0...group) {
                    var qh = g0 + gi;
                    var wgt = Mem.loadF32(scB + Usize.fromInt((qh * rowStride + l) * 4));
                    var ob = outB + Usize.fromInt((qh * headDim + b * 32) * 4);
                    var wv = SIMD4f.splat(wgt);
                    for (t in 0...8) {
                        var po = Ptr.fromRaw(ob + Usize.fromInt(t * 16));
                        var ov = SIMD4f.load(po);
                        var vv = SIMD4f.load(Ptr.fromRaw(vs + Usize.fromInt(t * 16)));
                        ov = ov.add(vv.mul(wv));
                        ov.store(po);
                    }
                }
            }
            l++;
        }
    }

}
