package nue;

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
 * Per-row INT8 weight × INT8 activation GEMM via VNNI — the x86/NUC analogue of
 * the Apple AMX f16 fast path. `y[m,n] = x[m,k] @ w[n,k]^T`, `w` quantized to
 * symmetric per-row int8.
 *
 * The activation is quantized to int8 then shifted `+128 → u8`, so
 * `SIMD4i32.dotI8U8` lowers to a single `vpdpbusd` (x86 `avx_vnni`) / SDOT (arm)
 * per 16 lanes — the JIT emits the instruction directly (`try_build_x86_vpdpbusd`),
 * no runtime FFI. A per-row `128·rowsum(w)` correction recovers the signed dot
 * from the unsigned-rhs product — the Q8 flash shifted-byte + block-sum trick.
 *
 * `k` must be a multiple of 16 (BERT: 384, 1536); no 256-element super-block
 * constraint (unlike Q4_K_M), so it fits the 384-dim encoder projections.
 *
 * Buffers cross the API as raw `Usize` addresses, NOT `Bytes` objects: a `Bytes`
 * passed as a cross-module parameter arrives with a bad backing (writes past the
 * first byte fault). The caller owns the `Bytes` and passes `.address()`.
 */
class Int8Matmul {
    /**
     * Quantize an F32 weight `[n,k]` to symmetric per-row int8. `qBase` (n·k
     * int8), `sBase` (n f32 scales = maxAbs/127), `mBase` (n i32 row sums) are
     * addresses of caller-owned buffers. Call once at load; reused per forward.
     */
    public static function quantizeWeight(w:Tensor, n:Int, k:Int, qBase:Usize, sBase:Usize, mBase:Usize):Void {
        var wBase = w.data().raw();
        for (i in 0...n) {
            var rowF = wBase + Usize.fromInt((i * k) << 2);
            var mAcc = SIMD4f.splat(0.0);
            var jv = 0;
            while (jv < k) {
                var vv = SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt(jv << 2)));
                mAcc = mAcc.max(vv.abs());
                jv += 4;
            }
            var maxAbs = mAcc.get(0);
            var la1 = mAcc.get(1); if (la1 > maxAbs) maxAbs = la1;
            var la2 = mAcc.get(2); if (la2 > maxAbs) maxAbs = la2;
            var la3 = mAcc.get(3); if (la3 > maxAbs) maxAbs = la3;
            var scale = maxAbs > 0.0 ? maxAbs / 127.0 : 1.0;
            var invD = maxAbs > 0.0 ? 127.0 / maxAbs : 0.0;
            Mem.storeF32(sBase + Usize.fromInt(i << 2), scale);
            var sum = 0;
            var qRow = qBase + Usize.fromInt(i * k);
            for (j in 0...k) {
                var v = Mem.loadF32(rowF + Usize.fromInt(j << 2)) * invD;
                var q = v >= 0.0 ? Std.int(v + 0.5) : Std.int(v - 0.5);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                Mem.storeU8(qRow + Usize.fromInt(j), q & 0xFF);
                sum += q;
            }
            Mem.storeI32(mBase + Usize.fromInt(i << 2), sum);
        }
    }

    /**
     * `y[m,n] = x[m,k] @ (int8 w)[n,k]^T`, dequantized to F32. `qwBase`/`wsBase`/
     * `wmBase` are the `quantizeWeight` output addresses. Returns a fresh `[m,n]`
     * F32 tensor. `k % 16 == 0`.
     */
    public static function matmul(x:Tensor, m:Int, k:Int, n:Int,
            qwBase:Usize, wsBase:Usize, wmBase:Usize, ?sp:SpinPool):Tensor {
        var y = Tensor.zeros([m, n], F32);
        var yBase = y.data().raw();
        var xBase = x.data().raw();

        // Per-call activation scratch: u8 quants [m*k] + f32 row scales [m].
        var ax = Bytes.alloc(m * k);
        var adS = Bytes.alloc(m * 4);
        var axBase = ax.address();
        var adBase = adS.address();
        for (t in 0...m) quantizeActRow(xBase, axBase, adBase, t, k);

        for (t in 0...m) {
            var xScale = Mem.loadF32(adBase + Usize.fromInt(t << 2));
            var aRow = axBase + Usize.fromInt(t * k);
            var yRow = yBase + Usize.fromInt((t * n) << 2);
            for (i in 0...n) {
                // raw = sum(w * (q_x + 128)) = signedDot + 128*rowsum(w).
                var raw = dotRow(qwBase + Usize.fromInt(i * k), aRow, k);
                var signedDot = raw - (Mem.loadI32(wmBase + Usize.fromInt(i << 2)) << 7);
                var v = xScale * Mem.loadF32(wsBase + Usize.fromInt(i << 2)) * signedDot;
                Mem.storeF32(yRow + Usize.fromInt(i << 2), v);
            }
        }
        ax.free();
        adS.free();
        return y;
    }

    /** One output cell: sum_kk dotI8U8(w[kk..], a[kk..]) over k lanes -> i32. */
    static function dotRow(wRow:Usize, aRow:Usize, k:Int):Int {
        var acc = SIMD4i32.splat(0);
        var kk = 0;
        while (kk < k) {
            var wV = SIMD16i8.load(Ptr.fromRaw(wRow + Usize.fromInt(kk)));
            var aV = SIMD16i8.load(Ptr.fromRaw(aRow + Usize.fromInt(kk)));
            // a = signed weight, b = u8-shifted activation -> vpdpbusd (x86).
            acc = SIMD4i32.dotI8U8(acc, wV, aV);
            kk += 16;
        }
        return acc.sum();
    }

    /** Quantize activation row `t` to int8, then shift `+128 → u8` (dotI8U8 rhs). */
    static function quantizeActRow(xBase:Usize, axBase:Usize, adBase:Usize, t:Int, k:Int):Void {
        var rowF = xBase + Usize.fromInt((t * k) << 2);
        var mAcc = SIMD4f.splat(0.0);
        var j = 0;
        while (j < k) {
            var v = SIMD4f.load(Ptr.fromRaw(rowF + Usize.fromInt(j << 2)));
            mAcc = mAcc.max(v.abs());
            j += 4;
        }
        var maxAbs = mAcc.get(0);
        var l1 = mAcc.get(1); if (l1 > maxAbs) maxAbs = l1;
        var l2 = mAcc.get(2); if (l2 > maxAbs) maxAbs = l2;
        var l3 = mAcc.get(3); if (l3 > maxAbs) maxAbs = l3;
        var invD = maxAbs > 0.0 ? 127.0 / maxAbs : 0.0;
        Mem.storeF32(adBase + Usize.fromInt(t << 2), maxAbs > 0.0 ? maxAbs / 127.0 : 1.0);
        var aRow = axBase + Usize.fromInt(t * k);
        for (jj in 0...k) {
            var v = Mem.loadF32(rowF + Usize.fromInt(jj << 2)) * invD;
            var q = v >= 0.0 ? Std.int(v + 0.5) : Std.int(v - 0.5);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            Mem.storeU8(aRow + Usize.fromInt(jj), (q + 128) & 0xFF);
        }
    }
}
