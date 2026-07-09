package nue.transformer;

import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.Bytes;
import rayzor.Mem;
import rayzor.Usize;

/**
 * Pure-Haxe Q8_0 KV cache — the guest-owned counterpart to the plugin's
 * [KvCacheQ8], with identical storage layout so the two are byte-comparable.
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

    // Persistent decode-attention scratch. Sized to maxSeqLen on first decode.
    public var scrQ:Bytes;
    public var scrQScale:Bytes;
    public var scrScores:Bytes;
    public var scrV:Bytes;

    // Precomputed per-block f32 scale (= f16ToF32 of the stored f16), one
    // per (row, kv-head, block), filled once at append.
    public var scaleF32:Bytes;
    /** Per-block signed sum of the 32 stored q8 lanes. Used by Q8 flash
        attention's shifted-query VNNI path:
        dot(k, q + 128) - 128 * sum(k) == dot(k, q). */
    public var sumI32:Bytes;
    public var blocksPerRow:Int;

    public function new(maxSeqLen:Int, numKvHeads:Int, headDim:Int) {
        this.maxSeqLen = maxSeqLen;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.headBytes = (headDim >> 5) * 34;
        this.rowBytes = numKvHeads * headBytes;
        this.blocksPerRow = numKvHeads * (headDim >> 5);
        // +4 trailing bytes: scratch cell for the f32<->bits roundtrip in
        // the f16 scale encode (appends are single-threaded).
        this.data = Bytes.alloc(maxSeqLen * rowBytes + 4);
        this.scaleF32 = Bytes.alloc(maxSeqLen * blocksPerRow * 4);
        this.sumI32 = Bytes.alloc(maxSeqLen * blocksPerRow * 4);
    }

    /** Lazily allocate the decode scratch, sized to maxSeqLen so it never
        reallocates as context grows. Idempotent. */
    public function ensureDecodeScratch(numQHeads:Int):Void {
        if (scrScores != null) return;
        var bph = headDim >> 5;
        scrQ = Bytes.alloc(numQHeads * headDim);
        scrQScale = Bytes.alloc(numQHeads * bph * 4);
        scrScores = Bytes.alloc(numQHeads * maxSeqLen * 4);
        scrV = Bytes.alloc(numKvHeads * 128);
    }

    public function free():Void {
        if (data != null) {
            data.free();
            data = null;
        }
        if (scaleF32 != null) {
            scaleF32.free();
            scaleF32 = null;
        }
        if (sumI32 != null) {
            sumI32.free();
            sumI32 = null;
        }
        if (scrScores != null) {
            scrQ.free();
            scrQScale.free();
            scrScores.free();
            scrV.free();
            scrScores = null;
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
        var sBase:Usize = src.data().raw();
        var dBase:Usize = data.address();
        var scBase:Usize = scaleF32.address();
        var sumBase:Usize = sumI32.address();
        var cell:Usize = dBase + Usize.fromInt(maxSeqLen * rowBytes);
        for (r in 0...n) {
            var srcRow:Usize = sBase + Usize.fromInt(r * numKvHeads * headDim * 4);
            var dstRow:Usize = dBase + Usize.fromInt((row + r) * rowBytes);
            var scRow:Usize = scBase + Usize.fromInt((row + r) * blocksPerRow * 4);
            var sumRow:Usize = sumBase + Usize.fromInt((row + r) * blocksPerRow * 4);
            for (b in 0...blocksPerRow) {
                quantBlock(srcRow + Usize.fromInt(b * 128),
                    dstRow + Usize.fromInt(b * 34),
                    scRow + Usize.fromInt(b * 4),
                    sumRow + Usize.fromInt(b * 4), cell);
            }
        }
        return row + n;
    }

    /** Dequantize rows 0..len into caller-allocated `out`
        (`[len, numKvHeads, headDim]` F32) — the prefill attention path. */
    public function dequantInto(len:Int, out:Tensor):Void {
        var oBase:Usize = out.data().raw();
        var dBase:Usize = data.address();
        var blocksPerRow = numKvHeads * (headDim >> 5);
        for (r in 0...len) {
            var srcRow:Usize = dBase + Usize.fromInt(r * rowBytes);
            var dstRow:Usize = oBase + Usize.fromInt(r * numKvHeads * headDim * 4);
            for (b in 0...blocksPerRow) {
                var blk:Usize = srcRow + Usize.fromInt(b * 34);
                var sc = f16ToF32(Mem.loadI32(blk) & 0xFFFF);
                var dst:Usize = dstRow + Usize.fromInt(b * 128);
                for (i in 0...32) {
                    var qv = (Mem.loadU8(blk + Usize.fromInt(2 + i)) << 24) >> 24;
                    Mem.storeF32(dst + Usize.fromInt(i * 4), sc * qv);
                }
            }
        }
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

    /** IEEE f32 -> f16 bits, round to nearest even. */
    static inline function f16FromF32(v:Float, cell:Usize):Int {
        Mem.storeF32(cell, v);
        var x:Int = Mem.loadI32(cell);
        var sign:Int = (x >>> 16) & 0x8000;
        var exp:Int = (x >>> 23) & 0xFF;
        var man:Int = x & 0x7FFFFF;
        if (exp == 0xFF) return sign | 0x7C00 | (man != 0 ? 0x200 : 0);
        var e:Int = exp - 112;
        if (e >= 0x1F) return sign | 0x7C00;
        if (e <= 0) {
            if (e < -10) return sign;
            var m:Int = man | 0x800000;
            var shift:Int = 14 - e;
            var half1:Int = (1 << (shift - 1)) - 1;
            var q:Int = (m + half1 + ((m >>> shift) & 1)) >>> shift;
            return sign | q;
        }
        var rounded:Int = man + 0xFFF + ((man >>> 13) & 1);
        var q2:Int = rounded >>> 13;
        var bits:Int = (e << 10) + q2;
        if (bits >= 0x7C00) return sign | 0x7C00;
        return sign | bits;
    }

    /** Quantize one 32-float group at `srcF32` into the 34-byte Q8_0
        block at `dst`. */
    static inline function quantBlock(srcF32:Usize, dst:Usize, scaleDst:Usize,
            sumDst:Usize, cell:Usize):Void {
        var maxAbs = 0.0;
        for (i in 0...32) {
            var v = Mem.loadF32(srcF32 + Usize.fromInt(i * 4));
            var nv = -v;
            var av = v > nv ? v : nv;
            maxAbs = av > maxAbs ? av : maxAbs;
        }
        var scale = maxAbs == 0.0 ? 0.0 : maxAbs / 127.0;
        var inv = scale == 0.0 ? 0.0 : 1.0 / scale;
        var bits = f16FromF32(scale, cell);
        Mem.storeU8(dst, bits & 0xFF);
        Mem.storeU8(dst + Usize.fromInt(1), (bits >>> 8) & 0xFF);
        Mem.storeF32(scaleDst, f16ToF32(bits));
        var sum = 0;
        for (i in 0...32) {
            var x = Mem.loadF32(srcF32 + Usize.fromInt(i * 4)) * inv;
            var q = x >= 0 ? Std.int(x + 0.5) : Std.int(x - 0.5);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            sum += q;
            Mem.storeU8(dst + Usize.fromInt(2 + i), q & 0xFF);
        }
        Mem.storeI32(sumDst, sum);
    }
}
