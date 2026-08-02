package rayzor.ds;

import rayzor.Usize;

/**
 * The Apple AMX/Accelerate GEMM, and nothing else.
 *
 * Pure compute — quantised weight decode, f32→f16 narrowing — belongs in Haxe
 * (`nue.Q4Matmul`). What cannot live there is Accelerate itself: AMX is only
 * reachable across the FFI boundary. Exposing the GEMM alone keeps that
 * boundary at exactly one call, instead of handing Rust the dequant too.
 *
 * `c[m,n] = a[m,k] · b[n,k]^T`, operands f16, accumulation f32 (hence the
 * `_nt` shape: b is stored row-major by output feature, i.e. transposed).
 * Returns false on non-macOS builds or if BNNS declines the shape, so callers
 * must keep a portable fallback.
 */
@:native("rayzor::ds::AmxGemm")
extern class AmxGemm {
    /**
     * @param a16  m*k f16 activations
     * @param b16  n*k f16 weights
     * @param cf32 m*n writable f32 output
     */
    @:native("rayzor_amx_gemm_f16")
    public static function gemmF16(m:Int, k:Int, n:Int, a16:Usize, b16:Usize, cf32:Usize):Bool;
}
