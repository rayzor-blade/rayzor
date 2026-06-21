// Phase 3 gate: pure-Haxe Q4_K_M × Q8_K dot kernel, bit-exact* vs the Rust
// scalar oracle (runtime-core/src/quant/q4_k_m.rs:vec_dot_q4_K_q8_K_scalar,
// golden values from its golden_gate::golden_q4_dot_for_haxe_gate test).
//
// This is the kernel that lets nue's quantized matmul leave per-block Rust FFI.
// It composes the primitives landed this session: SIMD16i8 nibble-unpack
// (AND 0x0F / USHR 4), the fused SIMD4i32.dot (→ SDOT), and Bytes.address()
// loading. The integer sdot is bit-exact (SIMD); the float fold runs in Haxe
// f64 vs the oracle's f32, so the result agrees to ~f32 precision (*not bit-
// identical by construction — Haxe Float is f64 — but well within argmax/
// sampling tolerance; France→Paris greedy is the end-to-end gate at Phase 4).
//
// Passes on BOTH native AND wasm — Bytes are guest-resident on wasm (the
// wasm_runner allocates the buffer in guest linear memory), so the same
// SIMD16i8.load path works in-guest. Phase 4 swaps the Bytes source for the
// QTensor weight pointer (also guest-resident) and drives the band loop.

import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Usize;
import rayzor.Bytes;

class Main {
    static function pow2(e:Int):Float {
        var r = 1.0;
        if (e >= 0) { for (i in 0...e) r *= 2.0; } else { for (i in 0...(-e)) r *= 0.5; }
        return r;
    }

    // IEEE binary16 raw bits -> f32 value. :Int annotations are REQUIRED — a
    // bitwise result in an inferred var does int-truncating float arithmetic.
    static function f16ToF32(bits:Int):Float {
        var sign:Int = (bits >> 15) & 1;
        var exp:Int = (bits >> 10) & 0x1F;
        var mant:Int = bits & 0x3FF;
        var sgn = (sign == 1) ? -1.0 : 1.0;
        if (exp == 0) return sgn * mant * 0.000000059604644775390625; // 2^-24
        if (exp == 31) return (mant == 0) ? sgn * 1e38 : 0.0;
        return sgn * (1.0 + mant / 1024.0) * pow2(exp - 15);
    }

    // Integer dot of one 32-element sub-block: weight nibbles (low or high of
    // the qs bytes at wByteOff) · activation i8 (at actByteOff). Two i8x16
    // chunks accumulate into one i32x4; sum() is the exact 32-wide integer dot.
    static function subDot(wBase:Usize, wByteOff:Int, isHi:Bool, aBase:Usize, actByteOff:Int):Int {
        var acc = SIMD4i32.splat(0);
        for (half in 0...2) {
            var wRaw = SIMD16i8.load(Ptr.fromRaw(wBase + Usize.fromInt(wByteOff + half * 16)));
            var wNib = isHi ? SIMD16i8.ushr(wRaw, 4) : SIMD16i8.and(wRaw, SIMD16i8.splat(0x0F));
            var aVec = SIMD16i8.load(Ptr.fromRaw(aBase + Usize.fromInt(actByteOff + half * 16)));
            acc = SIMD4i32.dot(acc, wNib, aVec); // a = nibble 0..15, b = activation i7
        }
        return acc.sum();
    }

    // Q4_K_M × Q8_K dot. weight = 144 raw bytes (d,dmin,scales[12],qs[128]);
    // act = 256 i8 activations; bsums[16]; xd = activation scale.
    static function q4Dot(weight:Bytes, act:Bytes, bsums:Array<Int>, xd:Float):Float {
        var dBits:Int = weight.get(0) | (weight.get(1) << 8);
        var dminBits:Int = weight.get(2) | (weight.get(3) << 8);
        var d = f16ToF32(dBits);
        var dmin = f16ToF32(dminBits);
        var acc = 0.0;
        for (s in 0...8) {
            var sc6:Int; var mn6:Int;
            if (s < 4) {
                sc6 = weight.get(4 + s) & 63;
                mn6 = weight.get(4 + s + 4) & 63;
            } else {
                var a:Int = weight.get(4 + s + 4);
                var b:Int = weight.get(4 + s - 4);
                var c:Int = weight.get(4 + s);
                sc6 = (a & 0x0F) | (((b >> 6) & 3) << 4);
                mn6 = ((a >> 4) & 0x0F) | (((c >> 6) & 3) << 4);
            }
            var subScale = d * sc6;
            var subMin = dmin * mn6;
            var p:Int = s >> 1;
            var isHi = (s & 1) == 1;
            var sdot = subDot(weight.address(), 16 + p * 32, isHi, act.address(), s * 32);
            var bsum32 = bsums[2 * s] + bsums[2 * s + 1];
            acc += subScale * sdot - subMin * bsum32;
        }
        return xd * acc;
    }

    static inline function close(a:Float, b:Float):Bool {
        var d = a - b;
        if (d < 0) d = -d;
        return d < 0.01; // f64-vs-f32 fold slack; integer dot is exact
    }

    static function setScales(w:Bytes, scales:Array<Int>):Void {
        for (i in 0...12) w.set(4 + i, scales[i]);
    }

    static function main() {
        // Case 1: d=1.0, dmin=0.5, qs[i]=i, act=(i%16)-8, bsums all -8, xd=0.01.
        var w1 = Bytes.alloc(144);
        w1.set(0, 0x00); w1.set(1, 0x3C); w1.set(2, 0x00); w1.set(3, 0x38);
        setScales(w1, [1, 2, 3, 4, 1, 1, 1, 1, 21, 22, 23, 24]);
        for (i in 0...128) w1.set(16 + i, i);
        var a1 = Bytes.alloc(256);
        for (i in 0...256) a1.set(i, (i % 16) - 8);
        var bs1 = new Array<Int>();
        for (s in 0...16) bs1.push(-8);
        var d1 = q4Dot(w1, a1, bs1, 0.01);

        // Case 2: varied d/dmin/scales, per-lane nibbles, NEGATIVE activations.
        var w2 = Bytes.alloc(144);
        w2.set(0, 0x66); w2.set(1, 0x2E); w2.set(2, 0x00); w2.set(3, 0x2C); // 0x2E66, 0x2C00
        setScales(w2, [67, 72, 141, 146, 2, 69, 72, 75, 231, 28, 65, 118]);
        for (i in 0...128) w2.set(16 + i, (i * 7 + 3) % 256);
        var a2 = Bytes.alloc(256);
        for (i in 0...256) a2.set(i, (i * 11 + 5) % 64 - 32);
        var bs2 = [-8, -72, 56, -8, -8, -72, 56, -8, -8, -72, 56, -8, -8, -72, 56, -8];
        var d2 = q4Dot(w2, a2, bs2, 0.0375);

        var ok = close(d1, 75.8399963379) && close(d2, -14.1860599518);
        if (ok) {
            Sys.println("PASS q4km-dot case1=" + d1 + " (~75.84) case2=" + d2 + " (~-14.186)");
        } else {
            Sys.println("FAIL case1=" + d1 + " (want 75.84) case2=" + d2 + " (want -14.186)");
        }
    }
}
