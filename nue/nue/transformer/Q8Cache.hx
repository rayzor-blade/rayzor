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
        // Locals along the address path are explicitly :Usize — an
        // unannotated extern-return through a field-loaded receiver chain
        // currently infers Float and turns pointer adds into fp adds
        // (silent-decay compiler bug, repro in this file's history).
        var sBase:Usize = src.data().raw();
        var dBase:Usize = data.address();
        var cell:Usize = dBase + Usize.fromInt(maxSeqLen * rowBytes);
        var blocksPerRow = numKvHeads * (headDim >> 5);
        for (r in 0...n) {
            var srcRow:Usize = sBase + Usize.fromInt(r * numKvHeads * headDim * 4);
            var dstRow:Usize = dBase + Usize.fromInt((row + r) * rowBytes);
            // Heads are contiguous in both source and destination, so the
            // whole row is one linear run of 32-float groups -> 34B blocks.
            for (b in 0...blocksPerRow) {
                quantBlock(srcRow + Usize.fromInt(b * 128),
                    dstRow + Usize.fromInt(b * 34), cell);
            }
        }
        return row + n;
    }

    /** Dequantize rows 0..len into caller-allocated `out`
        (`[len, numKvHeads, headDim]` F32) — the prefill attention path.
        Shaped as fill-into rather than allocate-and-return, and named
        distinctly from the plugin extern's `dequantView`: cross-module
        resolution currently fails on Haxe methods that RETURN an extern
        class (forward-ref stub), and same-name cross-class dispatch
        collapses to an unresolvable bare symbol. Tensor PARAMS resolve
        fine (same shape as `append`). */
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

    /** IEEE f32 -> f16 bits, round to nearest even (the storage rounding
        for block scales). `cell` is a 4-byte scratch address for the
        float -> bits reinterpret. Structured as straight-line early
        returns with branchless mantissa-carry: `(e<<10) + q2` lets a
        rounded-up mantissa overflow arithmetically into the exponent
        (0x400 carry), and a cross-branch `q2 = 0; e += 1` reassignment
        is exactly the conditional-int-reassign shape the MIR lowering
        currently decays to Float (stealLoop family) — verified live:
        scales just below powers of two came back as f32 bit patterns.
        Bit-exact vs IEEE RNE on a 120k-value sweep. */
    static inline function f16FromF32(v:Float, cell:Usize):Int {
        Mem.storeF32(cell, v);
        var x:Int = Mem.loadI32(cell);
        var sign:Int = (x >>> 16) & 0x8000;
        var exp:Int = (x >>> 23) & 0xFF;
        var man:Int = x & 0x7FFFFF;
        if (exp == 0xFF) return sign | 0x7C00 | (man != 0 ? 0x200 : 0);
        var e:Int = exp - 112; // rebase bias 127 -> 15
        if (e >= 0x1F) return sign | 0x7C00;
        if (e <= 0) {
            if (e < -10) return sign; // underflow to signed zero
            var m:Int = man | 0x800000;
            var shift:Int = 14 - e; // 14..24
            var half1:Int = (1 << (shift - 1)) - 1;
            var q:Int = (m + half1 + ((m >>> shift) & 1)) >>> shift;
            // a carry out of the subnormal mantissa lands on exponent 1 —
            // the bit pattern is already correct
            return sign | q;
        }
        var rounded:Int = man + 0xFFF + ((man >>> 13) & 1);
        var q2:Int = rounded >>> 13;
        var bits:Int = (e << 10) + q2; // mantissa carry rolls into exponent
        if (bits >= 0x7C00) return sign | 0x7C00;
        return sign | bits;
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
