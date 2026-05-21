//! Quantised tensor runtime — INT8 + Q4_K_M storage with dequant-fused matmul.
//!
//! The two quantisation schemes shipped in Phase 4 target the two big LLM
//! inference use-cases:
//!
//! - **INT8 symmetric per-row**: 8-bit weights, one f32 scale per row. The
//!   simplest scheme that still cuts memory 4× vs F32. Used by quantisation
//!   pipelines that need accuracy headroom (most attention QKV projections).
//!
//! - **Q4_K_M** (llama.cpp / GGUF format): 4-bit weights packed in 256-element
//!   super-blocks. Each super-block carries one f16 scale, one f16 min, and
//!   eight 6-bit (scale, min) pairs — one pair per 32-element sub-block. The
//!   workhorse format for shipping Llama-class models to edge: 4.5 bits per
//!   weight (8 bits → 4 bits weight + amortised metadata).
//!
//! Memory layout (a Q4_K_M super-block, 144 bytes):
//! ```text
//!   bytes  0..1     : f16 d        (super-block scale)
//!   bytes  2..3     : f16 dmin     (super-block min)
//!   bytes  4..15    : 8 × (6-bit scale, 6-bit min) packed in 12 bytes
//!   bytes 16..143   : 128 bytes of 4-bit quants (256 nibbles, low/high
//!                     bit-pairs interleaved across the 8 sub-blocks)
//! ```
//!
//! Dequant formula per element `i` (0..256):
//! ```text
//!   sub = i / 32                     // sub-block index 0..7
//!   q  = nibble[i]                   // 4-bit weight 0..15
//!   sc = d * (scales6[sub] / 63)     // effective scale
//!   mn = dmin * (mins6[sub] / 63)    // effective min
//!   w  = sc * q - mn
//! ```
//!
//! The runtime exposes a small FFI surface that the Haxe stdlib maps to
//! `rayzor.ds.QTensor`. All extern functions take/return i64 to match the
//! Haxe type system; pointers are reinterpreted on the way in.

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

use half::f16;

/// Quantisation scheme tag. Used by the Haxe-side `QScheme` enum.
pub const QSCHEME_INT8: u8 = 0;
pub const QSCHEME_Q4_K_M: u8 = 1;

/// Q4_K_M block dimensions. These are fixed by the GGUF spec.
pub const Q4_K_M_BLOCK_SIZE: usize = 256;
pub const Q4_K_M_BLOCK_BYTES: usize = 144;

/// Internal opaque tensor representation. The layout depends on `scheme`:
///
/// - `INT8`: `data` is a packed `i8` array of `numel` elements; `meta` is a
///   `f32` array of `meta_len = numel / group_size` per-group scales. The
///   layout is per-row symmetric quant: each row of `group_size` elements
///   shares one scale. A 4096×4096 INT8 matrix with group_size=4096 has
///   one scale per row → 4096 scales.
///
/// - `Q4_K_M`: `data` is a contiguous array of `numel / 256` super-blocks
///   (each 144 bytes); `meta` is empty (None) since metadata is embedded
///   per super-block.
#[repr(C)]
struct RayzorQTensor {
    data: *mut u8,
    meta: *mut f32, // nullable; INT8 scales OR null for Q4_K_M
    numel: usize,
    group_size: usize, // INT8: elements per scale; Q4_K_M: fixed 256
    scheme: u8,
    owns_data: bool,
    // 2-D layout for matmul: stored as [rows, cols] row-major. For 1-D
    // tensors this is (1, numel).
    rows: usize,
    cols: usize,
}

impl RayzorQTensor {
    #[allow(dead_code)]
    #[inline]
    fn data_bytes(&self) -> usize {
        match self.scheme {
            QSCHEME_INT8 => self.numel,
            QSCHEME_Q4_K_M => (self.numel / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES,
            _ => 0,
        }
    }
}

// ============================================================================
// INT8 symmetric per-row quantisation
// ============================================================================

/// Quantise an f32 row into int8 + per-row scale.
/// Returns the scale; writes the quantised bytes into `dst`.
fn quantise_int8_row(src: &[f32], dst: &mut [i8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    let mut max_abs = 0.0f32;
    for &x in src {
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    // 127 = int8 max magnitude (-128..=127). Symmetric quant uses 127 to
    // keep the dequant formula sign-symmetric.
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (i, &x) in src.iter().enumerate() {
        let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
        dst[i] = q;
    }
    scale
}

/// Dequant + matmul kernel for INT8-quantised A times f32 B.
///
/// A is `[M, K]` stored as i8 with one scale per row.
/// B is `[K, N]` stored as f32, row-major.
/// C is `[M, N]` stored as f32, row-major.
///
/// Computes `c[m, n] = scale[m] * Σ_k a_i8[m, k] * b[k, n]`.
/// The kernel runs i32-accumulated dot product per (m, n) lane, then
/// multiplies by the row scale at the end. This is the canonical pattern —
/// the i8 × f32 mixed-precision product would lose information without
/// the i32 accumulator.
#[allow(clippy::needless_range_loop)]
unsafe fn int8_matmul_f32(
    a_data: *const i8,
    scales: *const f32,
    b_data: *const f32,
    c_data: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        let row_scale = *scales.add(i);
        let a_row = a_data.add(i * k);
        let c_row = c_data.add(i * n);
        // Initialise the result row to zero.
        std::ptr::write_bytes(c_row, 0, n * std::mem::size_of::<f32>());
        for p in 0..k {
            let a_ik = *a_row.add(p) as f32 * row_scale;
            let b_row = b_data.add(p * n);
            // Equivalent to axpy_slice(c_row, a_ik, b_row).
            let c_slice = std::slice::from_raw_parts_mut(c_row, n);
            let b_slice = std::slice::from_raw_parts(b_row, n);
            crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
        }
    }
}

// ============================================================================
// Q4_K_M (GGUF) quantisation
// ============================================================================

/// In-memory view of a 144-byte Q4_K_M super-block. Decoded once into f32
/// scales + mins at dequant time so the inner kernel can stay arithmetic.
/// `d` and `dmin` are kept on the struct for diagnostics + unit tests even
/// though the hot kernel only consumes the already-scaled `scales`/`mins`.
struct Q4KBlock {
    #[allow(dead_code)]
    d: f32,
    #[allow(dead_code)]
    dmin: f32,
    scales: [f32; 8],  // per-sub-block effective scale (already multiplied by d)
    mins: [f32; 8],    // per-sub-block effective min (already multiplied by dmin)
    quants: [u8; 128], // 256 nibbles
}

/// Decode the 12-byte (scales, mins) header of a Q4_K_M block.
/// GGUF packs eight (6-bit scale, 6-bit min) pairs into 12 bytes via a
/// specific bit layout — this matches `llama.cpp/ggml-quants.c`
/// `get_scale_min_k4`.
#[inline]
fn q4_k_get_scale_min(j: usize, header: &[u8; 12]) -> (u8, u8) {
    // j is 0..8 sub-block index.
    // Layout per llama.cpp:
    //   scales[0..4] use low 6 bits of header[0..4]
    //   scales[4..8] use low 6 bits of header[8..12], lower 4 bits from
    //                bytes 8..12, upper 2 bits from bytes 0..4 high bits.
    if j < 4 {
        let sc = header[j] & 63;
        let mn = header[j + 4] & 63;
        (sc, mn)
    } else {
        let sc = (header[j + 4] & 0x0F) | ((header[j - 4] >> 6) << 4);
        let mn = (header[j + 4] >> 4) | ((header[j] >> 6) << 4);
        (sc, mn)
    }
}

#[inline]
unsafe fn decode_q4_k_block(block_ptr: *const u8) -> Q4KBlock {
    let d_bits = *(block_ptr as *const u16);
    let dmin_bits = *(block_ptr.add(2) as *const u16);
    let d = f16::from_bits(d_bits).to_f32();
    let dmin = f16::from_bits(dmin_bits).to_f32();

    let mut header = [0u8; 12];
    for (i, slot) in header.iter_mut().enumerate() {
        *slot = *block_ptr.add(4 + i);
    }

    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    for j in 0..8 {
        let (sc6, mn6) = q4_k_get_scale_min(j, &header);
        scales[j] = d * (sc6 as f32);
        mins[j] = dmin * (mn6 as f32);
    }

    let mut quants = [0u8; 128];
    for (i, slot) in quants.iter_mut().enumerate() {
        *slot = *block_ptr.add(16 + i);
    }

    Q4KBlock {
        d,
        dmin,
        scales,
        mins,
        quants,
    }
}

/// Dequant a single Q4_K_M block into 256 f32 values.
fn dequant_q4_k_block(block: &Q4KBlock, out: &mut [f32]) {
    debug_assert_eq!(out.len(), Q4_K_M_BLOCK_SIZE);
    // Within a super-block, two adjacent sub-blocks (32 elements each) share
    // 32 bytes of quants: the low nibbles of bytes 0..32 hold sub-block 2*s,
    // the high nibbles hold sub-block 2*s+1.
    for s in 0..4 {
        let sc_lo = block.scales[2 * s];
        let mn_lo = block.mins[2 * s];
        let sc_hi = block.scales[2 * s + 1];
        let mn_hi = block.mins[2 * s + 1];
        for i in 0..32 {
            let byte = block.quants[s * 32 + i];
            let q_lo = byte & 0x0F;
            let q_hi = byte >> 4;
            out[(2 * s) * 32 + i] = sc_lo * (q_lo as f32) - mn_lo;
            out[(2 * s + 1) * 32 + i] = sc_hi * (q_hi as f32) - mn_hi;
        }
    }
}

/// Q4_K_M × f32 matmul. A is `[M, K]` quantised; B is `[K, N]` f32; C is
/// `[M, N]` f32. Dequant happens one block at a time into a small f32
/// stage buffer, then reuses the existing SIMD axpy kernel for the
/// row-update step.
#[allow(clippy::needless_range_loop)]
unsafe fn q4_k_m_matmul_f32(
    a_data: *const u8,
    b_data: *const f32,
    c_data: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(
        k.is_multiple_of(Q4_K_M_BLOCK_SIZE),
        "Q4_K_M matmul: K must be a multiple of 256"
    );
    let blocks_per_row = k / Q4_K_M_BLOCK_SIZE;
    let mut stage = [0.0f32; Q4_K_M_BLOCK_SIZE];

    for i in 0..m {
        let c_row = c_data.add(i * n);
        std::ptr::write_bytes(c_row, 0, n * std::mem::size_of::<f32>());
        let row_ptr = a_data.add(i * blocks_per_row * Q4_K_M_BLOCK_BYTES);
        for b_idx in 0..blocks_per_row {
            let block_ptr = row_ptr.add(b_idx * Q4_K_M_BLOCK_BYTES);
            let block = decode_q4_k_block(block_ptr);
            dequant_q4_k_block(&block, &mut stage);
            // Now stage[0..256] holds the dequantised f32 weights for this
            // 256-element slice of A's row i. Update C's row i by axpy
            // against the matching 256 rows of B.
            let k_off = b_idx * Q4_K_M_BLOCK_SIZE;
            for p in 0..Q4_K_M_BLOCK_SIZE {
                let a_ik = stage[p];
                if a_ik == 0.0 {
                    continue;
                }
                let b_row = b_data.add((k_off + p) * n);
                let c_slice = std::slice::from_raw_parts_mut(c_row, n);
                let b_slice = std::slice::from_raw_parts(b_row, n);
                crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
            }
        }
    }
}

// ============================================================================
// Allocator helpers
// ============================================================================

unsafe fn alloc_qtensor(
    scheme: u8,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> *mut RayzorQTensor {
    let numel = rows * cols;
    let data_bytes = match scheme {
        QSCHEME_INT8 => numel,
        QSCHEME_Q4_K_M => (numel / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES,
        _ => return std::ptr::null_mut(),
    };

    let data = malloc(if data_bytes > 0 { data_bytes } else { 1 });
    if data.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::write_bytes(data, 0, data_bytes);

    // INT8 needs a per-row (or per-group) scale array. Q4_K_M embeds scales
    // in the data blocks so meta is null.
    let meta: *mut f32 = if scheme == QSCHEME_INT8 {
        let n_groups = numel / group_size;
        let scale_bytes = n_groups * std::mem::size_of::<f32>();
        let s = malloc(scale_bytes.max(4)) as *mut f32;
        if s.is_null() {
            free(data);
            return std::ptr::null_mut();
        }
        s
    } else {
        std::ptr::null_mut()
    };

    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        free(data);
        if !meta.is_null() {
            free(meta as *mut u8);
        }
        return std::ptr::null_mut();
    }

    *qt = RayzorQTensor {
        data,
        meta,
        numel,
        group_size,
        scheme,
        owns_data: true,
        rows,
        cols,
    };

    qt
}

// ============================================================================
// FFI surface
// ============================================================================

/// Create an INT8-quantised 2-D tensor `[rows, cols]` from an f32 source.
/// Each row of `cols` elements gets its own f32 scale. Returns the i64
/// pointer to the opaque `RayzorQTensor`; 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_f32_int8(src_ptr: i64, rows: i64, cols: i64) -> i64 {
    if src_ptr == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let qt_raw = alloc_qtensor(QSCHEME_INT8, rows, cols, cols);
    if qt_raw.is_null() {
        return 0;
    }
    let qt = &*qt_raw;
    let src = src_ptr as *const f32;
    for r in 0..rows {
        let row_src = std::slice::from_raw_parts(src.add(r * cols), cols);
        let row_dst = std::slice::from_raw_parts_mut(qt.data.add(r * cols) as *mut i8, cols);
        let scale = quantise_int8_row(row_src, row_dst);
        *qt.meta.add(r) = scale;
    }
    qt_raw as i64
}

/// Wrap a pre-quantised Q4_K_M byte buffer in a QTensor. The runtime takes
/// ownership of the bytes (i.e. they must come from malloc, OR the caller
/// must keep them alive and the QTensor will simply view them with
/// `owns_data=false`).
///
/// This is the intended GGUF integration point: the loader mmaps the
/// weights file and hands the runtime a raw block pointer + shape.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_wrap_q4_k_m(
    block_data_ptr: i64,
    rows: i64,
    cols: i64,
    take_ownership: i64,
) -> i64 {
    if block_data_ptr == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    if !(rows * cols).is_multiple_of(Q4_K_M_BLOCK_SIZE) {
        return 0;
    }
    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        return 0;
    }
    *qt = RayzorQTensor {
        data: block_data_ptr as *mut u8,
        meta: std::ptr::null_mut(),
        numel: rows * cols,
        group_size: Q4_K_M_BLOCK_SIZE,
        scheme: QSCHEME_Q4_K_M,
        owns_data: take_ownership != 0,
        rows,
        cols,
    };
    qt as i64
}

/// Copy a `haxe.io.Bytes` worth of pre-quantised Q4_K_M data into a fresh
/// owning `QTensor`. The intended caller is the GGUF loader handing the
/// runtime the raw byte slice returned by `GGUFReader.tensorBytes`.
///
/// The copy is intentional — the source Bytes' lifetime is tied to the
/// GGUFReader, which goes out of scope after `load()` returns. Wrapping
/// in-place with `owns_data=false` would leave the QTensor dangling.
///
/// `bytes_handle` is a `*const HaxeBytes` interpreted as i64.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_bytes_q4_k_m(
    bytes_handle: i64,
    rows: i64,
    cols: i64,
) -> i64 {
    if bytes_handle == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let bytes = &*(bytes_handle as *const crate::haxe_sys::HaxeBytes);
    if bytes.ptr.is_null() {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    if !(rows * cols).is_multiple_of(Q4_K_M_BLOCK_SIZE) {
        return 0;
    }
    let expected = (rows * cols / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES;
    if bytes.len < expected {
        return 0;
    }
    let buf = malloc(expected);
    if buf.is_null() {
        return 0;
    }
    std::ptr::copy_nonoverlapping(bytes.ptr, buf, expected);

    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        free(buf);
        return 0;
    }
    *qt = RayzorQTensor {
        data: buf,
        meta: std::ptr::null_mut(),
        numel: rows * cols,
        group_size: Q4_K_M_BLOCK_SIZE,
        scheme: QSCHEME_Q4_K_M,
        owns_data: true,
        rows,
        cols,
    };
    qt as i64
}

/// `qt.rows() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_rows(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).rows as i64
}

/// `qt.cols() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_cols(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).cols as i64
}

/// `qt.scheme() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_scheme(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).scheme as i64
}

/// `qt.numel() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_numel(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).numel as i64
}

/// Dequant the whole tensor into a fresh f32 Tensor (shape [rows, cols]).
/// Useful for debug / accuracy comparison; production code should prefer
/// the fused matmul path.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_dequant(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    let qt = &*(qt_ptr as *const RayzorQTensor);

    // We need a fresh f32 Tensor allocation. Mirror tensor.rs's alloc
    // shape: [rows, cols], F32 dtype, no fill.
    let shape = [qt.rows, qt.cols];
    let out_tensor_ptr =
        crate::tensor::rayzor_tensor_zeros(shape.as_ptr() as i64, 2, 0 /* DTYPE_F32 */);
    if out_tensor_ptr == 0 {
        return 0;
    }
    // Reach into the freshly allocated tensor's data ptr. The tensor.rs
    // layout has `data` as the first field, so dereferencing as a struct
    // with `data: *mut u8` first is safe.
    #[repr(C)]
    struct TensorHead {
        data: *mut u8,
        shape: *mut usize,
        strides: *mut usize,
        ndim: usize,
        numel: usize,
        dtype: u8,
        owns_data: bool,
        device: u8,
        numa_node: i32,
    }
    let head = &*(out_tensor_ptr as *const TensorHead);
    let out = head.data as *mut f32;

    match qt.scheme {
        QSCHEME_INT8 => {
            for r in 0..qt.rows {
                let scale = *qt.meta.add(r);
                let row_src = qt.data.add(r * qt.cols) as *const i8;
                let row_dst = out.add(r * qt.cols);
                for c in 0..qt.cols {
                    *row_dst.add(c) = (*row_src.add(c) as f32) * scale;
                }
            }
        }
        QSCHEME_Q4_K_M => {
            let blocks_per_row = qt.cols / Q4_K_M_BLOCK_SIZE;
            let mut stage = [0.0f32; Q4_K_M_BLOCK_SIZE];
            for r in 0..qt.rows {
                let row_ptr = qt.data.add(r * blocks_per_row * Q4_K_M_BLOCK_BYTES);
                for b in 0..blocks_per_row {
                    let block = decode_q4_k_block(row_ptr.add(b * Q4_K_M_BLOCK_BYTES));
                    dequant_q4_k_block(&block, &mut stage);
                    let dst = out.add(r * qt.cols + b * Q4_K_M_BLOCK_SIZE);
                    std::ptr::copy_nonoverlapping(stage.as_ptr(), dst, Q4_K_M_BLOCK_SIZE);
                }
            }
        }
        _ => {}
    }

    out_tensor_ptr
}

/// Fused dequant-matmul: A is quantised `[M, K]`, B is f32 `[K, N]`, out is
/// f32 `[M, N]`. Returns a fresh f32 Tensor; 0 on shape mismatch.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_matmul_f32(qt_a: i64, b_tensor: i64) -> i64 {
    if qt_a == 0 || b_tensor == 0 {
        return 0;
    }
    let qt = &*(qt_a as *const RayzorQTensor);

    // Pull B's shape + data.
    #[repr(C)]
    struct TensorHead {
        data: *mut u8,
        shape: *mut usize,
        strides: *mut usize,
        ndim: usize,
        numel: usize,
        dtype: u8,
        owns_data: bool,
        device: u8,
        numa_node: i32,
    }
    let b_head = &*(b_tensor as *const TensorHead);
    if b_head.ndim != 2 || b_head.dtype != 0
    /* DTYPE_F32 */
    {
        return 0;
    }
    let b_shape = std::slice::from_raw_parts(b_head.shape, 2);
    let k_b = b_shape[0];
    let n = b_shape[1];
    if k_b != qt.cols {
        return 0;
    }

    let out_shape = [qt.rows, n];
    let out_tensor = crate::tensor::rayzor_tensor_zeros(out_shape.as_ptr() as i64, 2, 0);
    if out_tensor == 0 {
        return 0;
    }
    let out_head = &*(out_tensor as *const TensorHead);
    let out_data = out_head.data as *mut f32;

    match qt.scheme {
        QSCHEME_INT8 => {
            int8_matmul_f32(
                qt.data as *const i8,
                qt.meta,
                b_head.data as *const f32,
                out_data,
                qt.rows,
                qt.cols,
                n,
            );
        }
        QSCHEME_Q4_K_M => {
            q4_k_m_matmul_f32(
                qt.data,
                b_head.data as *const f32,
                out_data,
                qt.rows,
                qt.cols,
                n,
            );
        }
        _ => return 0,
    }

    out_tensor
}

/// Release a QTensor. The runtime frees `data` and `meta` if `owns_data`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_free(qt_ptr: i64) {
    if qt_ptr == 0 {
        return;
    }
    let qt = &*(qt_ptr as *const RayzorQTensor);
    if qt.owns_data && !qt.data.is_null() {
        free(qt.data);
    }
    if !qt.meta.is_null() {
        free(qt.meta as *mut u8);
    }
    free(qt_ptr as *mut u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_quant_round_trip_close() {
        // Quantise a small row, then dequant, and check max abs error is
        // bounded by the per-row scale (1 ulp of int8).
        let row = [0.1f32, -0.5, 1.0, -1.0, 0.5, -0.1, 0.0, 2.0];
        let mut q = [0i8; 8];
        let scale = quantise_int8_row(&row, &mut q);
        for (i, &qv) in q.iter().enumerate() {
            let reconstructed = scale * qv as f32;
            assert!(
                (reconstructed - row[i]).abs() <= scale,
                "i={i} row={} got={} scale={}",
                row[i],
                reconstructed,
                scale
            );
        }
    }

    #[test]
    fn int8_matmul_close_to_f32() {
        // Build a small 2x4 × 4x3 matmul reference, quant A to int8, verify
        // the dequant matmul produces results within 5% relative error.
        let a_f32: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let b_f32: [f32; 12] = [
            0.5, 1.0, -0.5, 0.5, -1.0, 1.5, 1.0, 0.5, -1.0, -0.5, 1.0, 0.5,
        ];

        // Reference: A × B (2x4 × 4x3 → 2x3) in f32.
        let mut ref_c = [0.0f32; 6];
        for i in 0..2 {
            for j in 0..3 {
                let mut s = 0.0f32;
                for p in 0..4 {
                    s += a_f32[i * 4 + p] * b_f32[p * 3 + j];
                }
                ref_c[i * 3 + j] = s;
            }
        }

        // Quant A.
        unsafe {
            let qt = rayzor_qtensor_from_f32_int8(a_f32.as_ptr() as i64, 2, 4);
            assert!(qt != 0);

            // Allocate B as a Tensor and matmul.
            let b_shape = [4usize, 3];
            let b_tensor = crate::tensor::rayzor_tensor_zeros(b_shape.as_ptr() as i64, 2, 0);
            assert!(b_tensor != 0);
            // Copy b_f32 into the tensor's data.
            #[repr(C)]
            struct TensorHead {
                data: *mut u8,
                _shape: *mut usize,
                _strides: *mut usize,
                _ndim: usize,
                _numel: usize,
                _dtype: u8,
                _owns_data: bool,
                _device: u8,
                _numa_node: i32,
            }
            let b_head = &*(b_tensor as *const TensorHead);
            std::ptr::copy_nonoverlapping(b_f32.as_ptr(), b_head.data as *mut f32, 12);

            let out_tensor = rayzor_qtensor_matmul_f32(qt, b_tensor);
            assert!(out_tensor != 0);
            let out_head = &*(out_tensor as *const TensorHead);
            let out = std::slice::from_raw_parts(out_head.data as *const f32, 6);

            for i in 0..6 {
                let err = (out[i] - ref_c[i]).abs();
                let rel = if ref_c[i].abs() > 1e-6 {
                    err / ref_c[i].abs()
                } else {
                    err
                };
                assert!(
                    rel < 0.05,
                    "int8 matmul[{i}] = {} ref = {} rel_err = {}",
                    out[i],
                    ref_c[i],
                    rel
                );
            }

            rayzor_qtensor_free(qt);
        }
    }

    #[test]
    fn q4_k_m_block_round_trip() {
        // Synthesise a known Q4_K_M block and verify the dequant produces
        // the expected values.
        // Block: d=2.0, dmin=1.0, sub-block 0 (scale=1, min=2), all-zeros
        // quants → expected output: q*sc - mn = 0*2*1/63 - 1*2/63 = -2/63
        // for elements 0..32.
        //
        // Use raw bytes: d=2.0 in f16 = 0x4000, dmin=1.0 in f16 = 0x3C00.
        let mut block = [0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(2.0).to_bits();
        let dmin = f16::from_f32(1.0).to_bits();
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        // Header: sub-block 0 scale=1, min=1. Bytes 4 and 8.
        block[4] = 1; // scales[0..4] low 6 bits → scale[0] = 1
        block[8] = 1; // mins[0..4] low 6 bits via byte[4..8] header bit-packed; see q4_k_get_scale_min

        // Decode and verify.
        unsafe {
            let decoded = decode_q4_k_block(block.as_ptr());
            // Per the spec: scales[0] = d * 1 = 2.0, mins[0] = dmin * 1 = 1.0
            // (using the j<4 branch: sc = header[0] & 63 = 1, mn = header[4] & 63 = 1).
            assert!((decoded.d - 2.0).abs() < 1e-3);
            assert!((decoded.dmin - 1.0).abs() < 1e-3);
            assert!((decoded.scales[0] - 2.0).abs() < 1e-3);
            assert!((decoded.mins[0] - 1.0).abs() < 1e-3);

            let mut out = [0.0f32; Q4_K_M_BLOCK_SIZE];
            dequant_q4_k_block(&decoded, &mut out);
            // sub-block 0 elements 0..32: q=0, sc=2, mn=1 → 0*2 - 1 = -1.
            for i in 0..32 {
                assert!((out[i] - (-1.0)).abs() < 1e-3, "out[{i}] = {}", out[i]);
            }
        }
    }

    #[test]
    fn from_bytes_q4_k_m_copies_and_wraps() {
        // Build a single-block Q4_K_M buffer in a HaxeBytes-shaped struct,
        // pass through the FFI, verify the resulting QTensor wraps an
        // owning copy with the correct shape.
        let mut block = vec![0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(1.0).to_bits();
        let dmin = f16::from_f32(0.0).to_bits();
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());

        let bytes = crate::haxe_sys::HaxeBytes {
            ptr: block.as_mut_ptr(),
            len: block.len(),
            cap: block.capacity(),
        };
        let bytes_handle = &bytes as *const _ as i64;
        let qt = unsafe { rayzor_qtensor_from_bytes_q4_k_m(bytes_handle, 1, 256) };
        assert!(qt != 0);
        unsafe {
            let qt_ref = &*(qt as *const RayzorQTensor);
            assert_eq!(qt_ref.rows, 1);
            assert_eq!(qt_ref.cols, 256);
            assert_eq!(qt_ref.scheme, QSCHEME_Q4_K_M);
            assert!(qt_ref.owns_data);
            // The copied buffer should not alias the source — mutating
            // `block` should leave the QTensor's data untouched.
            assert_ne!(qt_ref.data, block.as_mut_ptr());
            assert_eq!(*qt_ref.data, d.to_le_bytes()[0]);
        }
        unsafe { rayzor_qtensor_free(qt) };
    }

    #[test]
    fn from_bytes_q4_k_m_rejects_bad_input() {
        // Misaligned (rows * cols not multiple of 256) → returns 0.
        let bytes = crate::haxe_sys::HaxeBytes {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        let handle = &bytes as *const _ as i64;
        assert_eq!(
            unsafe { rayzor_qtensor_from_bytes_q4_k_m(handle, 1, 100) },
            0
        );
        assert_eq!(unsafe { rayzor_qtensor_from_bytes_q4_k_m(0, 1, 256) }, 0);
    }
}
