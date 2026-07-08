package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Bytes;
import rayzor.Mem;
import rayzor.concurrent.SpinPool;

/**
 * Decode-time flash attention over the pure-Haxe Q8_0 KV cache ([Q8Cache]).
 *
 * One query row per q-head attends over `cacheLen` cached positions.
 * Work bands across kv-heads: a band owns its whole GQA group's score
 * and output rows outright, so bands share nothing but read-only cache
 * bytes — safe on the chunk-stealing pool.
 *
 * Per kv-head, the numerics are the standard max-shifted softmax in
 * three passes:
 *   1. scores[qh][l] = scale * (q[qh] . K[l,kvh]) — computed in the int
 *      domain: q rows are quantized to per-32-block q8 once per call, so
 *      each 32-wide sub-dot is one SIMD4i32.dot (SDOT) between the
 *      cached q8 block and the q row's q8 block; the cache blocks are
 *      never dequantized in this pass.
 *   2. softmax in place; weights are pre-multiplied by 1/denom so the V
 *      pass carries no division.
 *   3. out[qh] += w[l] * V[l,kvh] — each V block dequantizes to f32 once
 *      per (l, block) and feeds the whole GQA group; axpy is SIMD4f.
 */
class FlashDecode {
    // 0 = unread, 1 = on, 2 = off. Int (not a -1 sentinel): a cross-module
    // duplicate of this static starts at 0 and re-reads the env instead of
    // wrongly latching "off".
    static var _enabled:Int = 0;
    static var _batchMax:Int = 0;

    /** RAYZOR_HAXE_FLASH gates the pure-Haxe decode-attention path. */
    public static function enabled():Bool {
        if (_enabled == 0) {
            var v = Sys.getEnv("RAYZOR_HAXE_FLASH");
            _enabled = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _enabled == 1;
    }

    /** Max seqQ for the small-batch flash path. Large prefill still uses
        the GEMM path; speculative verification defaults to <=4 rows. */
    public static function batchMax():Int {
        if (_batchMax == 0) {
            var v = Sys.getEnv("RAYZOR_HAXE_FLASH_BATCH_MAX");
            var parsed = (v == null || v == "") ? 8 : Std.parseInt(v);
            if (parsed == null) parsed = 8;
            if (parsed < 1) parsed = 1;
            _batchMax = parsed;
        }
        return _batchMax;
    }

    static inline function f16ToF32(bits:Int):Float {
        var sign:Int = (bits >> 15) & 1;
        var exp:Int = (bits >> 10) & 0x1F;
        var mant:Int = bits & 0x3FF;
        var sgn = (sign == 1) ? -1.0 : 1.0;
        if (exp == 0) return sgn * mant * 0.000000059604644775390625;
        if (exp == 31) return (mant == 0) ? sgn * 1e38 : 0.0;
        return Mem.f32FromBits((sign << 31) | ((exp + 112) << 23) | (mant << 13));
    }

    /**
     * q: [1, numQHeads, headDim] F32 contiguous. Returns a fresh
     * [1, numQHeads, headDim] F32 tensor the caller frees.
     */
    public static function decode(kc:Q8Cache, vc:Q8Cache, q:Tensor, cacheLen:Int,
            numQHeads:Int, scale:Float, sp:Null<SpinPool>):Tensor {
        return decodeRange(kc, vc, q, 0, 1, cacheLen, numQHeads, scale, sp);
    }

    /**
     * Small-batch decode flash path for speculative verification. Query row
     * `qi` sees cache rows `[0, baseLen + qi]`, matching the causal mask after
     * the caller has already appended all draft K/V rows to the cache.
     */
    public static function decodeBatch(kc:Q8Cache, vc:Q8Cache, q:Tensor,
            baseLen:Int, seqQ:Int, numQHeads:Int, scale:Float,
            sp:Null<SpinPool>):Tensor {
        if (seqQ <= 0) return null;
        return decodeRange(kc, vc, q, baseLen, seqQ, 0, numQHeads, scale, sp);
    }

    static function decodeRange(kc:Q8Cache, vc:Q8Cache, q:Tensor,
            baseLen:Int, seqQ:Int, fixedCacheLen:Int, numQHeads:Int,
            scale:Float, sp:Null<SpinPool>):Tensor {
        var numKvHeads = kc.numKvHeads;
        var headDim = kc.headDim;
        var group = Std.int(numQHeads / numKvHeads);
        var bph = headDim >> 5; // 32-quant blocks per head
        var headBytes = kc.headBytes;
        var rowBytes = kc.rowBytes;
        // :Usize on every extern-returned address (see Q8Cache.append —
        // unannotated inference through field-loaded receivers decays to
        // Float and corrupts the pointer arithmetic).
        var kBase:Usize = kc.data.address();
        var vBase:Usize = vc.data.address();
        var qBase:Usize = q.data().raw();
        // Precomputed per-block f32 scales (filled once at append), read
        // zero-copy in the passes instead of re-decoding f16 per token per
        // cached position — O(cacheLen) redundancy that grows with context.
        var kScale:Usize = kc.scaleF32.address();
        var vScale:Usize = vc.scaleF32.address();
        var kSum:Usize = kc.sumI32.address();
        var blocksPerRow:Int = kc.blocksPerRow;

        var out = Tensor.zeros([seqQ, numQHeads, headDim], DType.F32);
        var outB:Usize = out.data().raw();

        // Persistent scratch owned by the K cache (allocated once, sized to
        // maxSeqLen). scores doubles as the softmax-weight buffer; its rows
        // are maxSeqLen-strided so a partially-filled context indexes the
        // same cells regardless of cacheLen. vScr holds one dequantized
        // 32-float V block per kv-head band.
        kc.ensureDecodeScratch(numQHeads, seqQ);
        var rowStride = kc.maxSeqLen;
        var qqB:Usize = kc.scrQ.address();
        var qscB:Usize = kc.scrQScale.address();
        var scB:Usize = kc.scrScores.address();
        var vScrB:Usize = kc.scrV.address();

        // Quantize the q rows to per-32-block q8 (seqQ*numQHeads*headDim floats).
        for (qi in 0...seqQ) {
            var qRowBase = qi * numQHeads;
            for (qh in 0...numQHeads) {
                var qRow = qRowBase + qh;
                for (b in 0...bph) {
                    var src = qBase + Usize.fromInt((qRow * headDim + b * 32) * 4);
                    var maxAbs = 0.0;
                    for (i in 0...32) {
                        var v = Mem.loadF32(src + Usize.fromInt(i * 4));
                        // abs via max(v,-v): reassigning v boxes the Float (see
                        // Q4Matmul.quantizeBlock).
                        var nv = -v;
                        var av = v > nv ? v : nv;
                        maxAbs = av > maxAbs ? av : maxAbs;
                    }
                    var s = maxAbs == 0.0 ? 0.0 : maxAbs / 127.0;
                    var inv = s == 0.0 ? 0.0 : 1.0 / s;
                    Mem.storeF32(qscB + Usize.fromInt((qRow * bph + b) * 4), s);
                    var dst = qqB + Usize.fromInt(qRow * headDim + b * 32);
                    for (i in 0...32) {
                        var x = Mem.loadF32(src + Usize.fromInt(i * 4)) * inv;
                        var r = x >= 0 ? Std.int(x + 0.5) : Std.int(x - 0.5);
                        if (r > 127) r = 127;
                        if (r < -128) r = -128;
                        // Store q+128. Pass 1 uses unsigned*signed VNNI as
                        // dot(K, q+128), then subtracts 128*sum(K) to recover
                        // signed Q8xQ8 exactly.
                        Mem.storeU8(dst + Usize.fromInt(i), (r + 128) & 0xFF);
                    }
                }
            }
        }

        var band = function(lo:Int, hi:Int, w:Int):Void {
            for (idx in lo...hi) {
                var qi = Std.int(idx / numKvHeads);
                var h = idx - qi * numKvHeads;
                var visibleLen = fixedCacheLen > 0 ? fixedCacheLen : baseLen + qi + 1;
                bandOne(h, qi * numQHeads, qi * numQHeads, idx, group, bph,
                    headBytes, rowBytes, visibleLen, rowStride,
                    headDim, scale, kBase, vBase, qqB, qscB, scB, vScrB, outB,
                    kScale, vScale, kSum, blocksPerRow);
            }
        };
        if (sp != null) {
            sp.parallelRows(seqQ * numKvHeads, band);
        } else {
            band(0, seqQ * numKvHeads, 0);
        }
        return out;
    }

    /** All three passes for one kv-head's GQA group. Touches only that
        group's score rows, output rows, and this band's V scratch slot. */
    static function bandOne(h:Int, qBaseHead:Int, outBaseHead:Int,
            vScratchHead:Int, group:Int, bph:Int, headBytes:Int,
            rowBytes:Int, cacheLen:Int, rowStride:Int, headDim:Int, scale:Float,
            kBase:Usize, vBase:Usize, qqB:Usize, qscB:Usize, scB:Usize,
            vScrB:Usize, outB:Usize, kScale:Usize, vScale:Usize,
            kSum:Usize, blocksPerRow:Int):Void {
        var g0 = h * group;
        var hb = h * bph; // this kv-head's first block index within a row

        // -- pass 1: scores = scale * (q . K), int-domain SDOT.
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
                    var qRow = qBaseHead + qh;
                    var qOff = qqB + Usize.fromInt(qRow * headDim + b * 32);
                    var acc = SIMD4i32.splat(0);
                    acc = SIMD4i32.dotI8U8(acc, k0, SIMD16i8.load(Ptr.fromRaw(qOff)));
                    acc = SIMD4i32.dotI8U8(acc, k1,
                        SIMD16i8.load(Ptr.fromRaw(qOff + Usize.fromInt(16))));
                    var qs = Mem.loadF32(qscB + Usize.fromInt((qRow * bph + b) * 4));
                    var p = ksc * qs * scale * (acc.sum() - 128 * ksum);
                    var cellA = scB + Usize.fromInt((qRow * rowStride + l) * 4);
                    if (b == 0) {
                        Mem.storeF32(cellA, p);
                    } else {
                        Mem.storeF32(cellA, Mem.loadF32(cellA) + p);
                    }
                }
            }
            l++;
        }

        // -- pass 2: softmax rows in place, weights pre-multiplied by 1/denom.
        for (gi in 0...group) {
            var qh = g0 + gi;
            var qRow = qBaseHead + qh;
            var rowA = scB + Usize.fromInt(qRow * rowStride * 4);
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

        // -- pass 3: out += w * V; V block dequantized once per (l, block)
        // and shared by the group. Output rows start zeroed (Tensor.zeros).
        var vs = vScrB + Usize.fromInt(vScratchHead * 128);
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
                    var qRow = qBaseHead + qh;
                    var outRow = outBaseHead + qh;
                    var wgt = Mem.loadF32(scB + Usize.fromInt((qRow * rowStride + l) * 4));
                    var ob = outB + Usize.fromInt((outRow * headDim + b * 32) * 4);
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
