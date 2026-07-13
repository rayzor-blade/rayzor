//! Apple Accelerate (AMX-backed BLAS) experiment surface — macOS only.
//!
//! This is a measurement/prototyping module, not (yet) wired into the model.
//! It exposes `cblas_sgemm` so we can benchmark the AMX matmul ceiling for the
//! model's shapes against the portable NEON path, and decide whether a
//! dequant-to-F32 + AMX route is worth the memory/bandwidth trade before
//! touching the hot kernels. The pure-Haxe / VNNI paths stay the portable
//! default; nothing here compiles off macOS.
#![cfg(target_os = "macos")]

use std::os::raw::c_int;

// CBLAS enums (Accelerate's cblas.h).
const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// C[m,n] = A[m,k] · B[k,n], all row-major, via Accelerate (AMX where present).
///
/// Panics on dimension/length mismatch. `a`/`b`/`c` are row-major contiguous.
pub fn sgemm_nn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k, "A must be m*k");
    assert_eq!(b.len(), k * n, "B must be k*n");
    assert_eq!(c.len(), m * n, "C must be m*n");
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            n as c_int,
            0.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

/// C[m,n] = A[m,k] · B[n,k]^T — the weight-transposed layout the decoder uses
/// (weights stored [out, in], activations [batch, in]). Accelerate handles the
/// transpose internally (no data movement).
pub fn sgemm_nt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k, "A must be m*k");
    assert_eq!(b.len(), n * k, "B (transposed) must be n*k");
    assert_eq!(c.len(), m * n, "C must be m*n");
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            k as c_int,
            0.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}
