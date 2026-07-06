package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.Bytes;
import rayzor.Mem;
import rayzor.Usize;

/**
 * Pure-Haxe Q8_0 KV cache — the guest-owned counterpart to the plugin's
 * [KvCacheQ8], with identical storage layout so the two are byte-comparable.
 *
 * Layout: `maxSeqLen` rows, each row `numKvHeads * headDim/32` blocks of
 * 34 bytes (2-byte f16 scale + 32 i8 quants), heads contiguous within a
 * row. ~3.76x smaller than F32.
 *
 * Quantization per 32-block: scale = maxAbs/127 (f32), quants =
 * round-half-away-from-zero(x/scale) clamped to [-128,127]. The stored
 * scale is the f16 rounding (to nearest even) of the f32 scale, but the
 * quants divide by the UNROUNDED f32 scale — same convention as the
 * plugin kernel, so blocks match it bit-for-bit on identical input.
 */
class Q8Cache {
    public var data:Bytes;
    public var maxSeqLen:Int;
    public var numKvHeads:Int;
    public var headDim:Int;
    /** Bytes per (row, head): (headDim/32) * 34. */
    public var headBytes:Int;
    /** Bytes per cache row: numKvHeads * headBytes. */
    public var rowBytes:Int;

    public function new(maxSeqLen:Int, numKvHeads:Int, headDim:Int) {
        this.maxSeqLen = maxSeqLen;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.headBytes = (headDim >> 5) * 34;
        this.rowBytes = numKvHeads * headBytes;
        // +4 trailing bytes: scratch cell for the f32<->bits roundtrip in
        // the f16 scale encode (appends are single-threaded).
        this.data = Bytes.alloc(maxSeqLen * rowBytes + 4);
    }

    public function free():Void {
        if (data != null) {
            data.free();
            data = null;
        }
    }

    /**
     * Quantize-append `src` ([n, numKvHeads, headDim] F32 contiguous)
     * starting at row `row`. Returns the new length (row + n), or -1 on
     * overflow/shape mismatch.
     */
    public function append(row:Int, src:Tensor):Int {
        var shp = src.shape();
        var n = shp[0];
        if (row + n > maxSeqLen) return -1;
        if (shp[1] != numKvHeads || shp[2] != headDim) return -1;
        var sBase = src.data().raw();
        var dBase = data.address();
        var cell = dBase + Usize.fromInt(maxSeqLen * rowBytes);
        var blocksPerRow = numKvHeads * (headDim >> 5);
        for (r in 0...n) {
            var srcRow = sBase + Usize.fromInt(r * numKvHeads * headDim * 4);
            var dstRow = dBase + Usize.fromInt((row + r) * rowBytes);
            // Heads are contiguous in both source and destination, so the
            // whole row is one linear run of 32-float groups -> 34B blocks.
            for (b in 0...blocksPerRow) {
                quantBlock(srcRow + Usize.fromInt(b * 128),
                    dstRow + Usize.fromInt(b * 34), cell);
            }
        }
        return row + n;
    }

    /** Dequantize rows 0..len into a fresh owning F32 tensor
        `[len, numKvHeads, headDim]` (the prefill attention path). */
    public function dequantView(len:Int):Tensor {
        var out = Tensor.zeros([len, numKvHeads, headDim], DType.F32);
        var oBase = out.data().raw();
        var dBase = data.address();
        var blocksPerRow = numKvHeads * (headDim >> 5);
        for (r in 0...len) {
            var srcRow = dBase + Usize.fromInt(r * rowBytes);
            var dstRow = oBase + Usize.fromInt(r * numKvHeads * headDim * 4);
            for (b in 0...blocksPerRow) {
                var blk = srcRow + Usize.fromInt(b * 34);
                var sc = f16ToF32(Mem.loadI32(blk) & 0xFFFF);
                var dst = dstRow + Usize.fromInt(b * 128);
                for (i in 0...32) {
                    var qv = (Mem.loadU8(blk + Usize.fromInt(2 + i)) << 24) >> 24;
                    Mem.storeF32(dst + Usize.fromInt(i * 4), sc * qv);
                }
            }
        }
        return out;
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

    /** IEEE f32 -> f16 bits, round to nearest even (the storage rounding
        for block scales). `cell` is a 4-byte scratch address for the
        float -> bits reinterpret. */
    static inline function f16FromF32(v:Float, cell:Usize):Int {
        Mem.storeF32(cell, v);
        var x = Mem.loadI32(cell);
        var sign = (x >>> 16) & 0x8000;
        var exp = (x >>> 23) & 0xFF;
        var man = x & 0x7FFFFF;
        var r:Int;
        if (exp == 0xFF) {
            // inf/nan (nan keeps a payload bit)
            r = sign | 0x7C00 | (man != 0 ? 0x200 : 0);
        } else {
            var e = exp - 112; // rebase bias 127 -> 15
            if (e >= 0x1F) {
                r = sign | 0x7C00;
            } else if (e <= 0) {
                if (e < -10) {
                    r = sign; // underflow to signed zero
                } else {
                    // subnormal: shift the implicit-1 mantissa into place
                    var m = man | 0x800000;
                    var shift = 14 - e; // 14..24
                    var q = m >>> shift;
                    var rem = m & ((1 << shift) - 1);
                    var half = 1 << (shift - 1);
                    if (rem > half || (rem == half && (q & 1) == 1)) q += 1;
                    // a carry out of the subnormal mantissa lands on
                    // exponent 1 — the bit pattern is already correct
                    r = sign | q;
                }
            } else {
                var q2 = man >>> 13;
                var rem2 = man & 0x1FFF;
                if (rem2 > 0x1000 || (rem2 == 0x1000 && (q2 & 1) == 1)) {
                    q2 += 1;
                    if (q2 == 0x400) {
                        q2 = 0;
                        e += 1;
                    }
                }
                r = (e >= 0x1F) ? (sign | 0x7C00) : (sign | (e << 10) | q2);
            }
        }
        return r;
    }

    /** Quantize one 32-float group at `srcF32` into the 34-byte Q8_0
        block at `dst`. */
    static inline function quantBlock(srcF32:Usize, dst:Usize, cell:Usize):Void {
        var maxAbs = 0.0;
        for (i in 0...32) {
            var v = Mem.loadF32(srcF32 + Usize.fromInt(i * 4));
            if (v < 0) v = -v;
            if (v > maxAbs) maxAbs = v;
        }
        var scale = maxAbs == 0.0 ? 0.0 : maxAbs / 127.0;
        var inv = scale == 0.0 ? 0.0 : 1.0 / scale;
        var bits = f16FromF32(scale, cell);
        Mem.storeU8(dst, bits & 0xFF);
        Mem.storeU8(dst + Usize.fromInt(1), (bits >>> 8) & 0xFF);
        for (i in 0...32) {
            var x = Mem.loadF32(srcF32 + Usize.fromInt(i * 4)) * inv;
            var q = x >= 0 ? Std.int(x + 0.5) : Std.int(x - 0.5);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            Mem.storeU8(dst + Usize.fromInt(2 + i), q & 0xFF);
        }
    }
}
