package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Mem;
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

    /** RAYZOR_HAXE_FLASH gates the pure-Haxe decode-attention path. */
    public static function enabled():Bool {
        if (_enabled == 0) {
            var v = Sys.getEnvOr("NUE_FLASH", "RAYZOR_HAXE_FLASH");
            _enabled = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
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
    public static function batchMax():Int {
        return 1;
    }

    public static function decodeBatch(kc:Q8Cache, vc:Q8Cache, q:Tensor,
            baseLen:Int, seqQ:Int, numQHeads:Int, scale:Float,
            sp:Null<SpinPool>):Tensor {
        return null;
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
