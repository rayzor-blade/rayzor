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

    /** RAYZOR_HAXE_FLASH gates the pure-Haxe decode-attention path. */
    public static function enabled():Bool {
        if (_enabled == 0) {
            var v = Sys.getEnv("RAYZOR_HAXE_FLASH");
            _enabled = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _enabled == 1;
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
        var numKvHeads = kc.numKvHeads;
        var headDim = kc.headDim;
        var group = Std.int(numQHeads / numKvHeads);
        var bph = headDim >> 5; // 32-quant blocks per head
        var headBytes = kc.headBytes;
        var rowBytes = kc.rowBytes;
        var kBase = kc.data.address();
        var vBase = vc.data.address();
        var qBase = q.data().raw();

        var out = Tensor.zeros([1, numQHeads, headDim], DType.F32);
        var outB = out.data().raw();

        // Per-call scratch. scores doubles as the softmax-weight buffer;
        // vScr holds one dequantized 32-float V block per kv-head band.
        var qq = Bytes.alloc(numQHeads * headDim);
        var qsc = Bytes.alloc(numQHeads * bph * 4);
        var scores = Bytes.alloc(numQHeads * cacheLen * 4);
        var vScr = Bytes.alloc(numKvHeads * 128);
        var qqB = qq.address();
        var qscB = qsc.address();
        var scB = scores.address();
        var vScrB = vScr.address();

        // Quantize the q rows to per-32-block q8 (numQHeads*headDim floats).
        for (qh in 0...numQHeads) {
            for (b in 0...bph) {
                var src = qBase + Usize.fromInt((qh * headDim + b * 32) * 4);
                var maxAbs = 0.0;
                for (i in 0...32) {
                    var v = Mem.loadF32(src + Usize.fromInt(i * 4));
                    if (v < 0) v = -v;
                    if (v > maxAbs) maxAbs = v;
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
                    Mem.storeU8(dst + Usize.fromInt(i), r & 0xFF);
                }
            }
        }

        var band = function(lo:Int, hi:Int, w:Int):Void {
            for (h in lo...hi) {
                bandOne(h, group, bph, headBytes, rowBytes, cacheLen, headDim,
                    scale, kBase, vBase, qqB, qscB, scB, vScrB, outB);
            }
        };
        if (sp != null) {
            sp.parallelRows(numKvHeads, band);
        } else {
            band(0, numKvHeads, 0);
        }
        return out;
    }

    /** All three passes for one kv-head's GQA group. Touches only that
        group's score rows, output rows, and this band's V scratch slot. */
    static function bandOne(h:Int, group:Int, bph:Int, headBytes:Int,
            rowBytes:Int, cacheLen:Int, headDim:Int, scale:Float,
            kBase:Usize, vBase:Usize, qqB:Usize, qscB:Usize, scB:Usize,
            vScrB:Usize, outB:Usize):Void {
        var g0 = h * group;

        // -- pass 1: scores = scale * (q . K), int-domain SDOT.
        var l = 0;
        while (l < cacheLen) {
            var rowP = kBase + Usize.fromInt(l * rowBytes + h * headBytes);
            for (b in 0...bph) {
                var blk = rowP + Usize.fromInt(b * 34);
                var ksc = f16ToF32(Mem.loadI32(blk) & 0xFFFF);
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
                    var cellA = scB + Usize.fromInt((qh * cacheLen + l) * 4);
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
            var rowA = scB + Usize.fromInt(qh * cacheLen * 4);
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
        var vs = vScrB + Usize.fromInt(h * 128);
        l = 0;
        while (l < cacheLen) {
            var rowP = vBase + Usize.fromInt(l * rowBytes + h * headBytes);
            for (b in 0...bph) {
                var blk = rowP + Usize.fromInt(b * 34);
                var vsc = f16ToF32(Mem.loadI32(blk) & 0xFFFF);
                for (i in 0...32) {
                    var qv = (Mem.loadU8(blk + Usize.fromInt(2 + i)) << 24) >> 24;
                    Mem.storeF32(vs + Usize.fromInt(i * 4), vsc * qv);
                }
                for (gi in 0...group) {
                    var qh = g0 + gi;
                    var wgt = Mem.loadF32(scB + Usize.fromInt((qh * cacheLen + l) * 4));
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
