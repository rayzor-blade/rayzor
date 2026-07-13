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

// ---- int8 x int8 -> int32 GEMM via BNNS (AMX integer path) ------------------
// Go/no-go surface: measure whether Accelerate's int8 matmul dispatches to AMX
// and beats the NEON SDOT baseline. Keeps the quantized representation (1 B/param
// vs F32's 4), the only AMX route worth shipping for a Q4 model. BNNSMatMul is
// __API_DEPRECATED (macOS 15, "use BNNSGraph") but still functional — fine for a
// measurement; a real integration would move to BNNSGraph.
// PROBED (accel_bench): int8 combos are REJECTED — the public matmul is
// float-only. f16 passes at 1.0-1.6 TF/s, so the shipping AMX route is
// f16 x f16 -> f32 below.

use std::os::raw::c_void;

// ---- f32 -> f16 bulk conversion via vImage (correct rounding + subnormals) --

#[repr(C)]
struct VImageBuffer {
    data: *mut c_void,
    height: usize, // vImagePixelCount
    width: usize,
    row_bytes: usize,
}

extern "C" {
    fn vImageConvert_PlanarFtoPlanar16F(
        src: *const VImageBuffer,
        dest: *const VImageBuffer,
        flags: u32,
    ) -> isize; // vImage_Error, 0 = success
}

/// Convert `src` f32 values to IEEE binary16 bits in `dst` (same length).
/// Chunked so a multi-hundred-MB weight never presents vImage a single
/// giant row. Returns false if vImage reports an error.
pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) -> bool {
    assert_eq!(src.len(), dst.len());
    const CHUNK: usize = 1 << 20;
    let mut off = 0;
    while off < src.len() {
        let n = CHUNK.min(src.len() - off);
        let sb = VImageBuffer {
            data: src[off..].as_ptr() as *mut c_void,
            height: 1,
            width: n,
            row_bytes: n * 4,
        };
        let db = VImageBuffer {
            data: dst[off..].as_mut_ptr() as *mut c_void,
            height: 1,
            width: n,
            row_bytes: n * 2,
        };
        let rc = unsafe { vImageConvert_PlanarFtoPlanar16F(&sb, &db, 0) };
        if rc != 0 {
            return false;
        }
        off += n;
    }
    true
}

#[repr(C)]
struct BnnsNdArrayDescriptor {
    flags: u32,
    layout: u32,
    size: [usize; 8],
    stride: [usize; 8],
    data: *mut c_void,
    data_type: u32,
    table_data: *mut c_void,
    table_data_type: u32,
    data_scale: f32,
    data_bias: f32,
}

const BNNS_DTYPE_INT8: u32 = 0x20000 | 8; // IntBit | 8
const BNNS_DTYPE_INT32: u32 = 0x20000 | 32; // IntBit | 32
                                            // RowMajorMatrix: value(row,col) at col*stride[0]+row*stride[1];
                                            // size[0]=cols, size[1]=rows. Contiguous row-major -> stride[0]=1, stride[1]=cols.
const BNNS_LAYOUT_ROW_MAJOR: u32 = 0x20000;

extern "C" {
    fn BNNSMatMul(
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        input_a: *const BnnsNdArrayDescriptor,
        input_b: *const BnnsNdArrayDescriptor,
        output: *const BnnsNdArrayDescriptor,
        workspace: *mut c_void,
        filter_params: *const c_void,
    ) -> i32;
}

fn desc_2d(rows: usize, cols: usize, data: *mut c_void, dtype: u32) -> BnnsNdArrayDescriptor {
    let mut d = BnnsNdArrayDescriptor {
        flags: 0,
        layout: BNNS_LAYOUT_ROW_MAJOR,
        size: [0; 8],
        stride: [0; 8],
        data,
        data_type: dtype,
        table_data: std::ptr::null_mut(),
        table_data_type: 0,
        data_scale: 1.0,
        data_bias: 0.0,
    };
    // RowMajorMatrix: size[0]=cols, size[1]=rows; contiguous row-major strides.
    d.size[0] = cols;
    d.size[1] = rows;
    d.stride[0] = 1;
    d.stride[1] = cols;
    d
}

const BNNS_DTYPE_F32: u32 = 0x10000 | 32; // FloatBit | 32
const BNNS_DTYPE_F16: u32 = 0x10000 | 16; // FloatBit | 16

/// f32 matmul via BNNSMatMul — ABI-isolation probe (is the call itself correct?).
pub fn matmul_f32_bnns_nt(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> bool {
    let da = desc_2d(m, k, a.as_ptr() as *mut c_void, BNNS_DTYPE_F32);
    let db = desc_2d(n, k, b.as_ptr() as *mut c_void, BNNS_DTYPE_F32);
    let dc = desc_2d(m, n, c.as_mut_ptr() as *mut c_void, BNNS_DTYPE_F32);
    let rc = unsafe {
        BNNSMatMul(
            false,
            true,
            1.0,
            &da,
            &db,
            &dc,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    rc == 0
}

/// f16 x f16 -> f16 matmul probe (AMX runs f16 at ~2x the f32 rate; f16
/// halves F32's bandwidth/residency cost). Bits are IEEE binary16 in u16.
pub fn matmul_f16_nt(m: usize, k: usize, n: usize, a: &[u16], b: &[u16], c: &mut [u16]) -> bool {
    let da = desc_2d(m, k, a.as_ptr() as *mut c_void, BNNS_DTYPE_F16);
    let db = desc_2d(n, k, b.as_ptr() as *mut c_void, BNNS_DTYPE_F16);
    let dc = desc_2d(m, n, c.as_mut_ptr() as *mut c_void, BNNS_DTYPE_F16);
    let rc = unsafe {
        BNNSMatMul(
            false,
            true,
            1.0,
            &da,
            &db,
            &dc,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    rc == 0
}

/// f16 inputs -> f32 output probe (wider accumulate/output).
pub fn matmul_f16f32_nt(m: usize, k: usize, n: usize, a: &[u16], b: &[u16], c: &mut [f32]) -> bool {
    let da = desc_2d(m, k, a.as_ptr() as *mut c_void, BNNS_DTYPE_F16);
    let db = desc_2d(n, k, b.as_ptr() as *mut c_void, BNNS_DTYPE_F16);
    let dc = desc_2d(m, n, c.as_mut_ptr() as *mut c_void, BNNS_DTYPE_F32);
    let rc = unsafe {
        BNNSMatMul(
            false,
            true,
            1.0,
            &da,
            &db,
            &dc,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    rc == 0
}

/// int8 inputs -> f32 output probe (int accumulate, dequant out).
pub fn matmul_i8f32_nt(m: usize, k: usize, n: usize, a: &[i8], b: &[i8], c: &mut [f32]) -> bool {
    let da = desc_2d(m, k, a.as_ptr() as *mut c_void, BNNS_DTYPE_INT8);
    let db = desc_2d(n, k, b.as_ptr() as *mut c_void, BNNS_DTYPE_INT8);
    let dc = desc_2d(m, n, c.as_mut_ptr() as *mut c_void, BNNS_DTYPE_F32);
    let rc = unsafe {
        BNNSMatMul(
            false,
            true,
            1.0,
            &da,
            &db,
            &dc,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    rc == 0
}

/// C[m,n] (i32) = A[m,k] (i8) · B[n,k] (i8)^T. Returns false if BNNS rejected it.
pub fn matmul_i8_nt(m: usize, k: usize, n: usize, a: &[i8], b: &[i8], c: &mut [i32]) -> bool {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);
    let da = desc_2d(m, k, a.as_ptr() as *mut c_void, BNNS_DTYPE_INT8);
    // B is [n,k]; transB folds it to [k,n] so C = A·B^T.
    let db = desc_2d(n, k, b.as_ptr() as *mut c_void, BNNS_DTYPE_INT8);
    let dc = desc_2d(m, n, c.as_mut_ptr() as *mut c_void, BNNS_DTYPE_INT32);
    let rc = unsafe {
        BNNSMatMul(
            false,
            true,
            1.0,
            &da,
            &db,
            &dc,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    rc == 0
}
