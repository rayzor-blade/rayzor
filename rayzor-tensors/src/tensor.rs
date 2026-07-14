//! Tensor runtime — N-dimensional array with shape, strides, and dtype.
//!
//! A Tensor is a heap-allocated struct containing:
//! - data: *mut f32 (or other dtype, but f32 is the primary path)
//! - shape: Vec<usize> stored as (ptr, len) inline
//! - strides: Vec<usize> stored as (ptr, len) inline
//! - dtype: u8 tag
//! - numel: usize
//! - ndim: usize
//! - rc: reference count for shared views
//!
//! At MIR/Haxe level, a Tensor is an opaque i64 (pointer).
//! All extern functions take/return i64 to match the type system.

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

// Pure-compute tensor helpers shared with `rayzor-runtime-wasm`.
use rayzor_runtime_core::tensor::{rms_norm, topk::recent_contains};

// =============================================================================
// Tensor-data alloc counters. The global TrackingAllocator (`#[global_allocator]`)
// only sees Rust's GlobalAlloc traffic; tensor data buffers go through
// `libc::malloc` directly (see `alloc_tensor` and friends), so they are
// INVISIBLE to that path. These counters close the gap — they're the only
// way to attribute the "Activity Monitor says 28 GB" mystery to a Haxe-side
// kernel choice.
//
// Dumped at exit when RAYZOR_DUMP_TENSOR_ALLOC=1 by the profile crate's
// atexit hook (which calls `rayzor_dump_tensor_alloc_stats` via FFI).
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering as MemOrdering};

pub static TENSOR_DATA_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_DATA_FREE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_DATA_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_DATA_FREE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_DATA_LIVE_PEAK: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_POOL_HITS: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_POOL_MISSES: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_FREE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
pub static TENSOR_FREE_REFCOUNT_NONZERO: AtomicU64 = AtomicU64::new(0);
// Note: pool push/eviction/byte counters live on
// `tensor_pool::PoolStats` (read via `tensor_pool::global().stats.snapshot()`)
// — not as standalone atomics here. The dump below surfaces them.

#[inline(always)]
fn record_data_alloc(bytes: usize) {
    let prev = TENSOR_DATA_ALLOC_BYTES.fetch_add(bytes as u64, MemOrdering::Relaxed);
    TENSOR_DATA_ALLOC_COUNT.fetch_add(1, MemOrdering::Relaxed);
    let live =
        (prev + bytes as u64).saturating_sub(TENSOR_DATA_FREE_BYTES.load(MemOrdering::Relaxed));
    TENSOR_DATA_LIVE_PEAK.fetch_max(live, MemOrdering::Relaxed);
}

#[inline(always)]
fn record_data_free(bytes: usize) {
    TENSOR_DATA_FREE_BYTES.fetch_add(bytes as u64, MemOrdering::Relaxed);
    TENSOR_DATA_FREE_COUNT.fetch_add(1, MemOrdering::Relaxed);
}

/// Dumps live tensor-data stats to stderr. Callable from Haxe via the
/// runtime mapping `rayzor_dump_tensor_alloc_stats` (no Haxe binding
/// today; invoked via the SIGTRAP/atexit hook in `profile.rs`).
#[no_mangle]
pub extern "C" fn rayzor_dump_tensor_alloc_stats() {
    let a = TENSOR_DATA_ALLOC_BYTES.load(MemOrdering::Relaxed);
    let f = TENSOR_DATA_FREE_BYTES.load(MemOrdering::Relaxed);
    let ac = TENSOR_DATA_ALLOC_COUNT.load(MemOrdering::Relaxed);
    let fc = TENSOR_DATA_FREE_COUNT.load(MemOrdering::Relaxed);
    let peak = TENSOR_DATA_LIVE_PEAK.load(MemOrdering::Relaxed);
    let hits = TENSOR_POOL_HITS.load(MemOrdering::Relaxed);
    let misses = TENSOR_POOL_MISSES.load(MemOrdering::Relaxed);
    let free_inv = TENSOR_FREE_INVOCATIONS.load(MemOrdering::Relaxed);
    let free_nz = TENSOR_FREE_REFCOUNT_NONZERO.load(MemOrdering::Relaxed);
    // Pool push/eviction/byte counters: read the LIVE counters from the
    // pool's own PoolStats (the previous standalone `TENSOR_POOL_PARKED`
    // / `TENSOR_POOL_EVICTED` atomics were declared but never
    // incremented — they always printed 0 regardless of pool activity,
    // which masked the fact that the pool defaults to OFF without
    // RZT_POOL=1).
    let pool = crate::tensor_pool::global();
    let pool_disabled = pool.is_disabled();
    let pool_snap = pool.stats.snapshot();
    let pushes = pool_snap.pushes;
    let evictions = pool_snap.evictions;
    let pool_cur_b = pool_snap.current_bytes as u64;
    let pool_peak_b = pool_snap.peak_bytes as u64;
    let hit_rate = if hits + misses > 0 {
        100.0 * hits as f64 / (hits + misses) as f64
    } else {
        0.0
    };
    let pool_state = if pool_disabled { "DISABLED" } else { "enabled" };
    eprintln!(
        "[tensor-data] pool={pool_state} allocs={ac} frees={fc} \
         alloc_bytes={a} ({a_mb:.1} MB) free_bytes={f} ({f_mb:.1} MB) \
         live={live} ({live_mb:.1} MB) peak={peak} ({peak_mb:.1} MB) \
         pool_hits={hits} pool_misses={misses} pool_hit_rate={hit_rate:.1}% \
         free_inv={free_inv} free_nonzero={free_nz} \
         pool_pushes={pushes} pool_evictions={evictions} \
         pool_current_bytes={pool_cur_b} pool_peak_bytes={pool_peak_b}",
        a_mb = a as f64 / 1_048_576.0,
        f_mb = f as f64 / 1_048_576.0,
        live = a.saturating_sub(f),
        live_mb = a.saturating_sub(f) as f64 / 1_048_576.0,
        peak_mb = peak as f64 / 1_048_576.0,
    );
    // Mirror to /tmp/rayzor-metrics-tensor.kv. The KV consumer in
    // `rayzor debug server` (src/debug/server.rs) reads `pool_hits`
    // and `pool_misses` only — the new `pool_pushes` / `pool_evictions`
    // / `pool_disabled` keys are additive and ignored by older readers.
    let kv = format!(
        "pool_disabled={pool_disabled}\nallocs={ac}\nfrees={fc}\n\
         alloc_bytes={a}\nfree_bytes={f}\npeak={peak}\n\
         pool_hits={hits}\npool_misses={misses}\nfree_inv={free_inv}\n\
         free_nonzero={free_nz}\npool_pushes={pushes}\npool_evictions={evictions}\n\
         pool_current_bytes={pool_cur_b}\npool_peak_bytes={pool_peak_b}\n"
    );
    let _ = std::fs::write("/tmp/rayzor-metrics-tensor.kv", kv);
}

// =============================================================================
// Alloc histogram (env-gated): writes one CSV line per `alloc_tensor` call.
// Enabled when `RZT_TENSOR_ALLOC_HISTOGRAM=1`. Output file controlled by
// `RZT_TENSOR_ALLOC_HISTOGRAM_PATH`, defaults to /tmp/alloc_hist.csv.
//
// This is for the tensor-pool design audit. Each line:
//   dtype,ndim,shape0,shape1,...
// =============================================================================

use std::io::Write as _;
use std::sync::Mutex;
use std::sync::OnceLock;

struct AllocHistogramSink {
    file: Option<std::fs::File>,
}

fn alloc_histogram_sink() -> &'static Option<Mutex<AllocHistogramSink>> {
    static SINK: OnceLock<Option<Mutex<AllocHistogramSink>>> = OnceLock::new();
    SINK.get_or_init(|| {
        match crate::env_var(
            "RZT_TENSOR_ALLOC_HISTOGRAM",
            "RAYZOR_TENSOR_ALLOC_HISTOGRAM",
        ) {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => {}
            _ => return None,
        }
        let path = crate::env_var(
            "RZT_TENSOR_ALLOC_HISTOGRAM_PATH",
            "RAYZOR_TENSOR_ALLOC_HISTOGRAM_PATH",
        )
        .unwrap_or_else(|_| "/tmp/alloc_hist.csv".to_string());
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .ok();
        Some(Mutex::new(AllocHistogramSink { file }))
    })
}

fn record_alloc_histogram(shape: &[usize], dtype: u8) {
    let Some(lock) = alloc_histogram_sink() else {
        return;
    };
    let Ok(mut sink) = lock.lock() else { return };
    let Some(file) = sink.file.as_mut() else {
        return;
    };
    let mut line = String::with_capacity(64);
    line.push_str(&dtype.to_string());
    line.push(',');
    line.push_str(&shape.len().to_string());
    for &d in shape {
        line.push(',');
        line.push_str(&d.to_string());
    }
    line.push('\n');
    let _ = file.write_all(line.as_bytes());
}

use half::{bf16, f16};

// DType tags matching the Haxe enum order.
// Stays in sync with `compiler/haxe-std/rayzor/ds/DType.hx` and the GPU-side
// constants in `gpu/src/buffer.rs`. Order is load-bearing: the Haxe enum
// produces ordinals 0..N matching these values, and MIR call sites pass
// the ordinal as the dtype arg.
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
pub const DTYPE_I32: u8 = 3;
pub const DTYPE_I8: u8 = 4;
pub const DTYPE_U8: u8 = 5;
pub const DTYPE_FP8_E4M3: u8 = 6;
pub const DTYPE_FP8_E5M2: u8 = 7;

// Device tags matching the Haxe Device enum.
// CPU(node) collapses node into the separate numa_node field on the struct
// — the tag itself is just "this lives on the CPU side". Vulkan and WebGPU
// both go through the `wgpu` crate at the runtime layer; the tag merely
// records which adapter family was selected.
pub(crate) const DEVICE_CPU: u8 = 0;
#[allow(dead_code)]
pub(crate) const DEVICE_METAL: u8 = 1;
#[allow(dead_code)]
pub(crate) const DEVICE_CUDA: u8 = 2;
#[allow(dead_code)]
pub(crate) const DEVICE_VULKAN: u8 = 3;
#[allow(dead_code)]
pub(crate) const DEVICE_WEBGPU: u8 = 4;

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        DTYPE_F16 => 2,
        DTYPE_BF16 => 2,
        DTYPE_I32 => 4,
        DTYPE_I8 => 1,
        DTYPE_U8 => 1,
        DTYPE_FP8_E4M3 => 1,
        DTYPE_FP8_E5M2 => 1,
        _ => 4, // default to f32
    }
}

// ============================================================================
// Per-element load / store helpers (storage ↔ f32)
//
// These convert one element at an arbitrary `*mut u8` data pointer between
// the storage representation (specified by the dtype tag) and f32 — the
// common compute representation. Phase 3b runs the kernels themselves in
// f32 and converts on load/store; Phase 3c specialises hot paths to native
// NEON f16 / x86 F16C where available.
//
// The fp8 paths (DTYPE_FP8_E4M3 / DTYPE_FP8_E5M2) implement a software
// dequant since neither the host CPU nor the `half` crate has built-in
// fp8 support. Quant-on-store is performed with naive round-to-nearest;
// kernels operating on fp8 weights are dequant-on-load only in Phase 3e.
// ============================================================================

#[inline(always)]
unsafe fn load_f32_at(data: *const u8, idx: usize, dtype: u8) -> f32 {
    match dtype {
        DTYPE_F32 => *(data as *const f32).add(idx),
        DTYPE_F16 => f16::from_bits(*(data as *const u16).add(idx)).to_f32(),
        DTYPE_BF16 => bf16::from_bits(*(data as *const u16).add(idx)).to_f32(),
        DTYPE_I32 => *(data as *const i32).add(idx) as f32,
        DTYPE_I8 => *(data as *const i8).add(idx) as f32,
        DTYPE_U8 => *data.add(idx) as f32,
        // FP8 reads go through a 256-entry LUT — ~16x faster than bit-twiddling.
        // See `init_fp8_luts` for the precomputed tables.
        DTYPE_FP8_E4M3 => FP8_E4M3_LUT[*data.add(idx) as usize],
        DTYPE_FP8_E5M2 => FP8_E5M2_LUT[*data.add(idx) as usize],
        _ => 0.0,
    }
}

#[inline(always)]
unsafe fn store_f32_at(data: *mut u8, idx: usize, dtype: u8, value: f32) {
    match dtype {
        DTYPE_F32 => *(data as *mut f32).add(idx) = value,
        DTYPE_F16 => *(data as *mut u16).add(idx) = f16::from_f32(value).to_bits(),
        DTYPE_BF16 => *(data as *mut u16).add(idx) = bf16::from_f32(value).to_bits(),
        DTYPE_I32 => *(data as *mut i32).add(idx) = value as i32,
        DTYPE_I8 => *(data as *mut i8).add(idx) = value as i8,
        DTYPE_U8 => *data.add(idx) = value as u8,
        DTYPE_FP8_E4M3 => *data.add(idx) = fp8_e4m3_from_f32(value),
        DTYPE_FP8_E5M2 => *data.add(idx) = fp8_e5m2_from_f32(value),
        _ => {}
    }
}

/// IEEE 754 FP8 E4M3 — 1 sign + 4 exponent + 3 mantissa bits, no infinity,
/// only one NaN encoding (0x7F / 0xFF). Bias = 7. Range ≈ ±448, ULP at 1.0
/// is 1/8. Common dequant target for INT4 → FP8 → FP16 pipelines.
///
/// Note: hot-path FP8 reads now go through `FP8_E4M3_LUT` (precomputed
/// 256-entry table). This function remains as the reference implementation
/// and is exercised by the LUT-construction const path below.
#[allow(dead_code)]
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 3) & 0xF;
    let mant = byte & 0x7;
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    let s = if sign == 0 { 1.0f32 } else { -1.0f32 };
    if exp == 0 {
        // subnormal
        s * (mant as f32) * (1.0 / 8.0) * 2.0f32.powi(-6)
    } else {
        let m = 1.0f32 + (mant as f32) / 8.0;
        s * m * 2.0f32.powi(exp as i32 - 7)
    }
}

fn fp8_e4m3_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    let abs = value.abs();
    if abs == 0.0 {
        return sign_bit;
    }
    // Clamp to E4M3 range (max = 448.0)
    let clamped = abs.min(448.0);
    let bits = clamped.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7FFFFF;

    if f32_exp < -6 {
        // Subnormal in E4M3 (or zero)
        // value = mant_e4m3 / 8 * 2^-6
        let scaled = (clamped / 2.0f32.powi(-6)) * 8.0;
        let m = scaled.round().clamp(0.0, 7.0) as u8;
        return sign_bit | m;
    }
    let exp_e4m3 = f32_exp + 7;
    if exp_e4m3 >= 0xF {
        // Saturate just below the NaN encoding
        return sign_bit | ((0xF) << 3) | 0x6;
    }
    // Round mantissa: f32 has 23 bits, E4M3 has 3.
    let m = (f32_mant >> 20) as u8;
    // Round-to-nearest-even on the discarded bits
    let lsb_mask = 1u32 << 20;
    let half_mask = 1u32 << 19;
    let round_bits = f32_mant & ((1u32 << 20) - 1);
    let rounded =
        if round_bits > half_mask || (round_bits == half_mask && (f32_mant & lsb_mask) != 0) {
            m.saturating_add(1)
        } else {
            m
        };
    if rounded > 7 {
        // mantissa overflow → bump exponent
        let new_exp = (exp_e4m3 as u8) + 1;
        if new_exp >= 0xF {
            return sign_bit | (0xF << 3) | 0x6;
        }
        sign_bit | (new_exp << 3)
    } else {
        sign_bit | ((exp_e4m3 as u8) << 3) | rounded
    }
}

/// IEEE 754 FP8 E5M2 — 1 sign + 5 exponent + 2 mantissa bits, supports inf
/// and standard NaN encodings. Bias = 15. Range ≈ ±57344, ULP at 1.0 = 1/4.
///
/// Hot-path FP8 E5M2 reads go through `FP8_E5M2_LUT`; this remains as the
/// reference for the LUT-build const path.
#[allow(dead_code)]
fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x3;
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }
    if exp == 0x1F {
        if mant == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return f32::NAN;
    }
    let s = if sign == 0 { 1.0f32 } else { -1.0f32 };
    if exp == 0 {
        s * (mant as f32) * (1.0 / 4.0) * 2.0f32.powi(-14)
    } else {
        let m = 1.0f32 + (mant as f32) / 4.0;
        s * m * 2.0f32.powi(exp as i32 - 15)
    }
}

fn fp8_e5m2_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    if value.is_infinite() {
        return sign_bit | (0x1F << 2);
    }
    let abs = value.abs();
    if abs == 0.0 {
        return sign_bit;
    }
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7FFFFF;
    if f32_exp < -14 {
        let scaled = (abs / 2.0f32.powi(-14)) * 4.0;
        let m = scaled.round().clamp(0.0, 3.0) as u8;
        return sign_bit | m;
    }
    let exp_e5m2 = f32_exp + 15;
    if exp_e5m2 >= 0x1F {
        return sign_bit | (0x1F << 2); // saturate to ±inf
    }
    let m = (f32_mant >> 21) as u8 & 0x3;
    let half_mask = 1u32 << 20;
    let round_bits = f32_mant & ((1u32 << 21) - 1);
    let rounded =
        if round_bits > half_mask || (round_bits == half_mask && (f32_mant >> 21) & 0x1 != 0) {
            m.saturating_add(1)
        } else {
            m
        };
    if rounded > 3 {
        let new_exp = (exp_e5m2 as u8) + 1;
        if new_exp >= 0x1F {
            return sign_bit | (0x1F << 2);
        }
        sign_bit | (new_exp << 2)
    } else {
        sign_bit | ((exp_e5m2 as u8) << 2) | rounded
    }
}

/// Precomputed 256-entry lookup tables for FP8 → f32. Generated once via
/// the const-eval-friendly construction below — both fp8 formats only have
/// 256 possible byte values, so the entire dequant is a single load.
const FP8_E4M3_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e4m3_to_f32_const(i as u8);
        i += 1;
    }
    t
};

const FP8_E5M2_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e5m2_to_f32_const(i as u8);
        i += 1;
    }
    t
};

/// const-friendly versions of the fp8 decoders — same arithmetic, but
/// expressed with operations that work inside `const fn` (no .powi, no
/// transcendentals, no float arithmetic that hits the runtime softfloat).
/// Computed at compile-time as part of `FP8_*_LUT` construction.
const fn fp8_e4m3_to_f32_const(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 3) & 0xF;
    let mant = byte & 0x7;
    if exp == 0 && mant == 0 {
        // Const fn can't express -0.0 literal cleanly; LUT consumer treats
        // both as zero so this is fine.
        return 0.0;
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    // value = sign * mantissa * 2^(exp - bias)
    // mantissa = (exp==0 ? mant/8 : 1 + mant/8)
    // bias = 7
    let mant_num = if exp == 0 {
        mant as u32
    } else {
        8 + mant as u32
    };
    let exp_val = if exp == 0 { -9i32 } else { exp as i32 - 10 };
    let magnitude = const_pow2(mant_num as f32, exp_val);
    if sign == 0 {
        magnitude
    } else {
        -magnitude
    }
}

const fn fp8_e5m2_to_f32_const(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x3;
    if exp == 0 && mant == 0 {
        return 0.0;
    }
    if exp == 0x1F {
        if mant == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return f32::NAN;
    }
    let mant_num = if exp == 0 {
        mant as u32
    } else {
        4 + mant as u32
    };
    let exp_val = if exp == 0 { -16i32 } else { exp as i32 - 17 };
    let magnitude = const_pow2(mant_num as f32, exp_val);
    if sign == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// `n * 2^k` in const context. Bit-manipulates the f32 representation
/// directly so we avoid runtime float ops (which const-eval forbids on
/// stable). Only valid for finite `n` and normal-range `k`.
const fn const_pow2(n: f32, k: i32) -> f32 {
    if n == 0.0 {
        return 0.0;
    }
    let bits = n.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let new_exp = exp_bits + k;
    if new_exp <= 0 {
        // Underflow to zero — close enough for LUT semantics.
        return 0.0;
    }
    if new_exp >= 0xFF {
        return f32::INFINITY;
    }
    let mant_and_sign = bits & 0x807FFFFF;
    f32::from_bits(mant_and_sign | ((new_exp as u32) << 23))
}

/// Bulk-fill `numel` elements at `data` with the given f32 value, in the
/// storage representation specified by `dtype`. Used by ones/full/zeros.
#[inline]
unsafe fn fill_dtype(data: *mut u8, numel: usize, dtype: u8, value: f32) {
    if value == 0.0 {
        // Every supported storage format encodes +0.0 as all zero bytes.
        std::ptr::write_bytes(data, 0, numel * dtype_size(dtype));
        return;
    }
    for i in 0..numel {
        store_f32_at(data, i, dtype, value);
    }
}

/// Internal tensor representation
#[repr(C)]
struct RayzorTensor {
    data: *mut u8,
    shape: *mut usize,
    strides: *mut usize,
    ndim: usize,
    numel: usize,
    dtype: u8,
    // `owns_data` indicates this wrapper owns its `data` / `shape` / `strides`
    // allocations and is responsible for freeing them. As of the clone-compact
    // fix (`rayzor_tensor_clone` always compacts to contiguous), `owns_data ==
    // true` ALSO implies canonical row-major strides AND
    // `data_bytes == numel * dtype_size(dtype)`. Every producer of an owning
    // tensor (`alloc_tensor`, dequant constructors, `rayzor_tensor_clone`)
    // upholds this invariant — view producers (`permute`, `slice`, `transpose*`,
    // `reshape`-aliased) set `owns_data=false` and may carry arbitrary strides.
    // A clone of a view compacts via a strided gather so the result is safe to
    // pool-admit. See memory bugs_clone_view_passthrough_invariant.md (FIXED).
    owns_data: bool,
    // Device placement. `device` is the device tag (DEVICE_CPU/DEVICE_METAL/
    // DEVICE_CUDA/DEVICE_WEBGPU). `numa_node` is meaningful only when
    // device == DEVICE_CPU: -1 means "no affinity hint", >= 0 names a NUMA
    // node from rayzor.concurrent.NumaTopology. Phase 1a default: every
    // existing allocation tags itself CPU/-1.
    device: u8,
    numa_node: i32,
    // Phase 1 ARC refcount. Every freshly-constructed wrapper starts at 1.
    // `rayzor_tensor_arc_clone` does a Relaxed fetch_add and returns the same
    // pointer (i64-ABI preserved). `rayzor_tensor_free` does AcqRel fetch_sub:
    // only the decrement-to-zero actually releases shape/strides/data/wrapper
    // (or pool-routes the owning slot). For views, `parent` is non-null and
    // points at the wrapper whose `data` we alias; the dec-to-zero path also
    // decrements the parent's refcount (recursively, in case of view-of-view).
    //
    // ABI note: the i64 handle is still a raw `*mut RayzorTensor`. Refcount
    // bookkeeping is purely a runtime-side detail; no Arc-into-raw / from-raw
    // round-tripping at FFI boundaries, so every `&*(t as *const RayzorTensor)`
    // pattern in the rest of the file still works unchanged.
    refcount: std::sync::atomic::AtomicUsize,
    // Null for owning tensors. Non-null for views — points at the wrapper
    // whose data buffer we alias. Held with one refcount bump on that parent;
    // released (Acquire fence + dec) when this view's own refcount hits zero.
    parent: *mut RayzorTensor,
}

impl RayzorTensor {
    /// Compute row-major strides from shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![0usize; ndim];
        strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Compute flat offset from multi-dimensional indices
    fn offset(&self, indices: &[usize]) -> usize {
        let strides = unsafe { std::slice::from_raw_parts(self.strides, self.ndim) };
        let mut off = 0usize;
        for i in 0..self.ndim {
            off += indices[i] * strides[i];
        }
        off
    }

    /// True iff this tensor's strides match the canonical row-major layout for
    /// its shape (i.e. `strides[i] == prod(shape[i+1..])`). After the clone-
    /// compact fix, `owns_data == true` does imply contiguity, but view
    /// producers (`permute`, `slice`, `transpose*`) carry arbitrary strides
    /// with `owns_data=false` — kernels that want a stride-1 fast path on a
    /// possibly-view input must still call this helper rather than gating on
    /// `owns_data`. See memory bugs_clone_view_passthrough_invariant.md.
    ///
    /// 0-D tensors are trivially contiguous. Length-1 axes are contiguous for
    /// any stride (no element is reached through them), matching numpy's
    /// `is_contiguous` semantics.
    unsafe fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let shape = std::slice::from_raw_parts(self.shape, self.ndim);
        let strides = std::slice::from_raw_parts(self.strides, self.ndim);
        let mut expected: usize = 1;
        for i in (0..self.ndim).rev() {
            if shape[i] != 1 && strides[i] != expected {
                return false;
            }
            expected *= shape[i];
        }
        true
    }
}

// ============================================================================
// Tensor pool integration
//
// `alloc_tensor` first asks `tensor_pool` for a recycled wrapper of the same
// `(dtype, shape)` class. On a hit the four mallocs (data, shape, strides,
// wrapper) all collapse to zero — the popped wrapper still holds its original
// buffers, and we just refill `data` per the `fill` arg. View producers
// (`reshape`-contiguous, `permute`, `slice`, `transpose`, `transpose_last2`)
// MUST bypass the pool: their wrappers carry `owns_data=false` and aliased
// `data` belonging to a parent. `rayzor_tensor_free` enforces this on the
// push side by routing only `owns_data=true` tensors through the pool.
// ============================================================================

use crate::tensor_pool::{self, PoolKey, PooledEntry, ShapeBuf};

/// Canonical FreeFn that `tensor_pool` invokes on eviction or drain. The
/// `PooledEntry::ptr` is the original `*mut RayzorTensor` (cast to `*mut u8`);
/// the freer reconstructs the wrapper, releases its data/shape/strides if
/// `owns_data`, then frees the wrapper itself. The `qtensor_meta_*` fields
/// must be null for plain tensors (set by QTensor's parallel free path).
unsafe fn tensor_pool_freer(entry: PooledEntry) {
    if entry.ptr.is_null() {
        return;
    }
    // Defensive: a plain-tensor pool entry should never carry qtensor meta.
    // If it ever does (bug elsewhere) we still want to release it rather
    // than leak.
    if !entry.qtensor_meta_ptr.is_null() {
        free(entry.qtensor_meta_ptr);
    }
    let t = &*(entry.ptr as *const RayzorTensor);
    if t.owns_data && !t.data.is_null() {
        let bytes = t.numel * dtype_size(t.dtype);
        free(t.data);
        record_data_free(bytes);
    }
    if !t.shape.is_null() {
        free(t.shape as *mut u8);
    }
    if !t.strides.is_null() {
        free(t.strides as *mut u8);
    }
    free(entry.ptr);
}

/// Compute the data-buffer byte count for a pooled tensor of `(dtype, shape)`.
/// Matches the formula used by `alloc_tensor` (numel * dtype_size).
#[inline]
fn pool_alloc_bytes(shape: &[usize], dtype: u8) -> usize {
    let numel: usize = shape.iter().product();
    numel * dtype_size(dtype)
}

/// Allocate a new tensor struct on the heap, return as i64.
///
/// Pool-first: tries `tensor_pool::global().try_pop()` for the requested
/// shape class; on a hit, the popped wrapper's data buffer is refilled
/// per `fill` and the same wrapper handle is returned (zero mallocs).
/// On a miss falls through to the canonical four-malloc path below.
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
unsafe fn alloc_tensor_with_zero_policy(
    shape: &[usize],
    dtype: u8,
    fill: Option<f32>,
    zero_unfilled: bool,
) -> i64 {
    // Env-gated allocation histogram for tensor-pool design audit. One CSV
    // line per call: dtype,ndim,shape0,shape1,...
    record_alloc_histogram(shape, dtype);
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let elem_size = dtype_size(dtype);
    let data_bytes = numel * elem_size;

    // ---- Pool fast path ----
    let key = PoolKey::from_shape(dtype, shape);
    if let Some(entry) = tensor_pool::global().try_pop(key, shape) {
        TENSOR_POOL_HITS.fetch_add(1, MemOrdering::Relaxed);
        // The wrapper, data, shape, strides are all reused. The shape vec is
        // already correct (matched in try_pop via the bucket-walk shape check)
        // and the strides for a given shape are deterministic (row-major) —
        // after `rayzor_tensor_clone`'s view-passthrough fix the pool never
        // sees non-canonical strides on `owns_data=true` tensors. We still
        // rewrite the strides here as defence-in-depth: the cost is `ndim`
        // pointer-stores (≤ 10 for every shape Llama ever produces) which is
        // negligible next to the data zero-fill below.
        let tensor = entry.ptr as *mut RayzorTensor;
        let t = &mut *tensor;
        // Reset the data buffer per the requested fill semantics. The
        // popped buffer is the original `numel * elem_size` block. Historical
        // alloc_tensor(fill=None) zeroes for callers that expect clean output;
        // full-overwrite kernels can opt out through alloc_tensor_uninit().
        if !t.data.is_null() && data_bytes > 0 {
            if let Some(val) = fill {
                if val == 0.0 {
                    std::ptr::write_bytes(t.data, 0, data_bytes);
                } else {
                    fill_dtype(t.data, numel, dtype, val);
                }
            } else if zero_unfilled {
                std::ptr::write_bytes(t.data, 0, data_bytes);
            }
        }
        // Defensive strides rewrite: compute canonical row-major strides
        // from `shape` and stamp them into the popped wrapper. The previous
        // owner's strides MUST already match this layout (the pool only
        // accepts `owns_data=true` entries and clone-of-view returns a
        // fresh malloc not pool-routed) — debug builds assert this.
        let canonical_strides = RayzorTensor::compute_strides(shape);
        if !t.strides.is_null() {
            debug_assert_eq!(
                std::slice::from_raw_parts(t.strides, ndim),
                canonical_strides.as_slice(),
                "pool-hit strides drifted from canonical row-major for shape {:?}",
                shape
            );
            for i in 0..ndim {
                *t.strides.add(i) = canonical_strides[i];
            }
        }
        // Refresh device tagging — the wrapper inherits whatever the prior
        // owner set; reset to the default so callers aren't surprised.
        t.device = DEVICE_CPU;
        t.numa_node = -1;
        // owns_data MUST be true on the way out; this should already hold
        // because we only push owning tensors into the pool.
        t.owns_data = true;
        // Phase 1 refcount reset: the previous owner reached refcount=0 and
        // pushed; the new owner starts at 1. `parent` is always null for
        // pool-hits because the pool only admits owning (non-view) wrappers.
        t.refcount.store(1, std::sync::atomic::Ordering::Relaxed);
        t.parent = std::ptr::null_mut();
        return tensor as i64;
    }

    // ---- Slow path: 4 mallocs ----

    TENSOR_POOL_MISSES.fetch_add(1, MemOrdering::Relaxed);

    // Allocate data
    let data = malloc(if data_bytes > 0 { data_bytes } else { 1 });
    if data.is_null() {
        return 0;
    }
    record_data_alloc(data_bytes);

    // Fill data
    if let Some(val) = fill {
        fill_dtype(data, numel, dtype, val);
    } else if zero_unfilled {
        std::ptr::write_bytes(data, 0, data_bytes);
    }

    // Allocate shape array
    let shape_ptr = malloc(ndim * std::mem::size_of::<usize>()) as *mut usize;
    if shape_ptr.is_null() {
        free(data);
        return 0;
    }
    for i in 0..ndim {
        *shape_ptr.add(i) = shape[i];
    }

    // Compute and allocate strides
    let strides = RayzorTensor::compute_strides(shape);
    let strides_ptr = malloc(ndim * std::mem::size_of::<usize>()) as *mut usize;
    if strides_ptr.is_null() {
        free(data);
        free(shape_ptr as *mut u8);
        return 0;
    }
    for i in 0..ndim {
        *strides_ptr.add(i) = strides[i];
    }

    // Allocate tensor struct
    let tensor = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if tensor.is_null() {
        free(data);
        free(shape_ptr as *mut u8);
        free(strides_ptr as *mut u8);
        return 0;
    }

    *tensor = RayzorTensor {
        data,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim,
        numel,
        dtype,
        owns_data: true,
        device: DEVICE_CPU,
        numa_node: -1,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
    };

    tensor as i64
}

#[inline]
unsafe fn alloc_tensor(shape: &[usize], dtype: u8, fill: Option<f32>) -> i64 {
    alloc_tensor_with_zero_policy(shape, dtype, fill, true)
}

#[inline]
unsafe fn alloc_tensor_uninit(shape: &[usize], dtype: u8) -> i64 {
    alloc_tensor_with_zero_policy(shape, dtype, None, false)
}

// ============================================================================
// Construction
// ============================================================================

// ============================================================================
// Plugin ABI accessors
//
// Stable cross-plugin surface for inspecting a `RayzorTensor` handle without
// the plugin needing to know the struct layout. Mirrors the shape declared
// in `rayzor_plugin::host_abi`. Bumping `rayzor_plugin::ABI_VERSION` is
// required whenever any of these change shape.
// ============================================================================

/// Plugin ABI: read a tensor's data pointer. Returns null for the
/// null handle (0) to keep plugin callers from segfaulting on a
/// missed null check.
#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_data(t: i64) -> *mut u8 {
    if t == 0 {
        return std::ptr::null_mut();
    }
    (*(t as *const RayzorTensor)).data
}

/// Plugin ABI: read a tensor's dtype tag. Returns 255 (an unused
/// dtype slot) for the null handle so plugins can sentinel-check
/// without UB.
#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_dtype(t: i64) -> u8 {
    if t == 0 {
        return u8::MAX;
    }
    (*(t as *const RayzorTensor)).dtype
}

/// Plugin ABI: read a tensor's ndim. Returns 0 for the null handle.
#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_ndim(t: i64) -> u32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const RayzorTensor)).ndim as u32
}

/// Plugin ABI: read a tensor's shape pointer. Returns null for the
/// null handle.
#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_shape(t: i64) -> *const usize {
    if t == 0 {
        return std::ptr::null();
    }
    (*(t as *const RayzorTensor)).shape
}

/// Plugin ABI: 1 if the tensor's strides match row-major
/// contiguous layout for its current shape, 0 otherwise. Returns 0
/// for the null handle.
#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_is_contiguous(t: i64) -> u8 {
    if t == 0 {
        return 0;
    }
    if (*(t as *const RayzorTensor)).is_contiguous() {
        1
    } else {
        0
    }
}

/// Plugin ABI: allocate a zero-initialised tensor with the given
/// shape + dtype. Mirror of `rayzor_tensor_zeros` but takes
/// `*const usize` directly so plugin code can pass a Rust slice
/// without an i64 cast. Returns 0 on shape rejection / OOM.
#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_alloc_zeros(
    shape_ptr: *const usize,
    ndim: usize,
    dtype: u8,
) -> i64 {
    if shape_ptr.is_null() || ndim == 0 {
        return 0;
    }
    let shape = std::slice::from_raw_parts(shape_ptr, ndim).to_vec();
    alloc_tensor(&shape, dtype, Some(0.0))
}

/// Tensor.zeros(shape_ptr: i64, ndim: i64, dtype: i64) -> i64
///
/// shape_ptr is a pointer to an array of i64 shape values (from Haxe Array<Int>).
/// We read ndim elements, convert to usize, and create the tensor.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_zeros(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8, Some(0.0))
}

/// Tensor.uninit(shape_ptr: i64, ndim: i64, dtype: i64) -> i64
///
/// Allocate an owning contiguous tensor without initialising its data buffer.
/// This is only valid for full-overwrite producers. General callers must use
/// Tensor.zeros/full so stale pooled bytes never become observable.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_uninit(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor_uninit(&shape, dtype as u8)
}

/// Tensor.ones(shape_ptr, ndim, dtype) -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ones(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8, Some(1.0))
}

/// Tensor.full(shape_ptr, ndim, value, dtype) -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_full(
    shape_ptr: i64,
    ndim: i64,
    value: f64,
    dtype: i64,
) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8, Some(value as f32))
}

/// Tensor.fromArray(data_ptr, data_len, dtype) -> i64
/// Creates a 1-D tensor with shape=[data_len] from a flat array of f64 values.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_array(data_ptr: i64, data_len: i64, dtype: i64) -> i64 {
    let numel = data_len as usize;
    let shape = vec![numel];
    let dtype_u8 = dtype as u8;

    let tensor_ptr = alloc_tensor(&shape, dtype_u8, None);
    if tensor_ptr == 0 {
        return 0;
    }

    let tensor = &*(tensor_ptr as *const RayzorTensor);

    // Copy f64 data from Haxe Array<Float>, converting to target dtype.
    // Goes through the generic store_f32_at helper so every supported
    // dtype (F32 / F16 / BF16 / I32 / I8 / U8 / FP8_E4M3 / FP8_E5M2) is
    // populated with the right storage format.
    let src = data_ptr as *const f64;
    for i in 0..numel {
        store_f32_at(tensor.data, i, dtype_u8, *src.add(i) as f32);
    }

    tensor_ptr
}

/// Materialise a fresh f32 Tensor from raw F16 bytes laid out in row-major
/// order with the given shape. Used by the GGUF loader for GGML dtype=1
/// (F16) tensors. The input is a `haxe.io.Bytes` whose underlying buffer
/// holds `numel * 2` bytes of little-endian IEEE 754 half-precision.
///
/// Bytes are interpreted as f16, widened to f32, and stored — i.e. the
/// output tensor is plain F32. Keeping it F32 sidesteps the half-kernel
/// gap (Phase 3 is partial: storage works, compute kernels don't).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_f16(
    bytes_handle: i64,
    shape_ptr: i64,
    ndim: i64,
) -> i64 {
    if bytes_handle == 0 {
        return 0;
    }
    let bytes = &*(bytes_handle as *const crate::haxe_sys::HaxeBytes);
    if bytes.ptr.is_null() {
        return 0;
    }
    let shape = read_shape(shape_ptr, ndim as usize);
    let numel: usize = shape.iter().product();
    if bytes.len < numel * 2 {
        return 0;
    }

    let tensor_ptr = alloc_tensor(&shape, DTYPE_F32, None);
    if tensor_ptr == 0 {
        return 0;
    }
    let tensor = &*(tensor_ptr as *const RayzorTensor);
    let dst = tensor.data as *mut f32;
    let src = bytes.ptr;
    for i in 0..numel {
        let lo = *src.add(i * 2) as u16;
        let hi = *src.add(i * 2 + 1) as u16;
        let bits = lo | (hi << 8);
        *dst.add(i) = half::f16::from_bits(bits).to_f32();
    }
    tensor_ptr
}

/// Materialise a fresh F32 Tensor from raw F32 bytes laid out row-major.
/// Bypasses the `Array<Float>` round-trip used by Tensor.fromArray, which
/// loses precision when crossing the array_push wrapper that takes the
/// element as `Any` (i64). For GGUF F32 tensors the bytes are already
/// little-endian f32, so we just memcpy them into a freshly allocated
/// tensor buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_f32(
    bytes_handle: i64,
    shape_ptr: i64,
    ndim: i64,
) -> i64 {
    if bytes_handle == 0 {
        return 0;
    }
    let bytes = &*(bytes_handle as *const crate::haxe_sys::HaxeBytes);
    if bytes.ptr.is_null() {
        return 0;
    }
    let shape = read_shape(shape_ptr, ndim as usize);
    let numel: usize = shape.iter().product();
    if bytes.len < numel * 4 {
        return 0;
    }
    let tensor_ptr = alloc_tensor(&shape, DTYPE_F32, None);
    if tensor_ptr == 0 {
        return 0;
    }
    let tensor = &*(tensor_ptr as *const RayzorTensor);
    std::ptr::copy_nonoverlapping(bytes.ptr, tensor.data, numel * 4);
    tensor_ptr
}

/// Materialise a fresh f32 Tensor from raw GGML Q8_0 bytes laid out in
/// 34-byte blocks (one f16 scale + 32 i8 weights). Used by the GGUF
/// loader for GGML dtype=8 (Q8_0) tensors.
///
/// As with F16: output is F32 to avoid needing a Q8_0-aware compute
/// kernel. Block-Q8_0 is uncommon in Q4_K_M models so the load-time
/// expansion cost is modest.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_q8_0(
    bytes_handle: i64,
    shape_ptr: i64,
    ndim: i64,
) -> i64 {
    if bytes_handle == 0 {
        return 0;
    }
    let bytes = &*(bytes_handle as *const crate::haxe_sys::HaxeBytes);
    if bytes.ptr.is_null() {
        return 0;
    }
    let shape = read_shape(shape_ptr, ndim as usize);
    let numel: usize = shape.iter().product();
    if !numel.is_multiple_of(32) {
        return 0;
    }
    let n_blocks = numel / 32;
    let expected = n_blocks * 34;
    if bytes.len < expected {
        return 0;
    }

    let tensor_ptr = alloc_tensor(&shape, DTYPE_F32, None);
    if tensor_ptr == 0 {
        return 0;
    }
    let tensor = &*(tensor_ptr as *const RayzorTensor);
    let dst = tensor.data as *mut f32;
    let src = bytes.ptr;
    for b in 0..n_blocks {
        let base = src.add(b * 34);
        let lo = *base as u16;
        let hi = *base.add(1) as u16;
        let scale = half::f16::from_bits(lo | (hi << 8)).to_f32();
        let q_base = base.add(2) as *const i8;
        let out_base = dst.add(b * 32);
        for j in 0..32 {
            *out_base.add(j) = (*q_base.add(j)) as f32 * scale;
        }
    }
    tensor_ptr
}

/// Tensor.rand(shape_ptr, ndim, dtype) -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rand(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    let tensor_ptr = alloc_tensor(&shape, dtype as u8, None);
    if tensor_ptr == 0 {
        return 0;
    }

    let tensor = &*(tensor_ptr as *const RayzorTensor);

    // Simple LCG random for deterministic "random" init
    if tensor.dtype == DTYPE_F32 {
        let dst = tensor.data as *mut f32;
        let mut seed: u64 = 0xDEADBEEF_CAFEBABE;
        for i in 0..tensor.numel {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((seed >> 33) as u32) & 0x7FFFFF; // 23 bits mantissa
            let val = (bits as f32) / (0x800000 as f32); // [0, 1)
            *dst.add(i) = val;
        }
    }

    tensor_ptr
}

// ============================================================================
// Properties
// ============================================================================

/// tensor.ndim() -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ndim(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.ndim as i64
}

/// tensor.numel() -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_numel(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.numel as i64
}

/// tensor.device() -> i64 (returns device tag: 0=CPU, 1=Metal, 2=Cuda, 3=WebGPU)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_device(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return DEVICE_CPU as i64;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.device as i64
}

/// tensor.numa_node() -> i64 (NUMA node hint when device == CPU; -1 means "any")
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_numa_node(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return -1;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.numa_node as i64
}

/// tensor.dtype() -> i64 (returns dtype tag)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_dtype(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.dtype as i64
}

/// tensor.shape() -> i64 (returns pointer to a heap-allocated HaxeArray of Int)
///
/// Allocates a HaxeArray struct + data buffer, copies shape dims as i64 values.
/// HaxeArray layout: { ptr: *mut u8, len: usize, cap: usize, elem_size: usize }
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    let ndim = t.ndim;
    let shape_slice = std::slice::from_raw_parts(t.shape, ndim);

    // Allocate HaxeArray struct (4 fields x 8 bytes = 32 bytes)
    let arr_ptr = malloc(32) as *mut usize;
    if arr_ptr.is_null() {
        return 0;
    }

    // Allocate data buffer for ndim i64 elements
    let elem_size = std::mem::size_of::<i64>();
    let cap = ndim.max(8); // match HaxeArray INITIAL_CAPACITY
    let data_ptr = malloc(cap * elem_size);
    if data_ptr.is_null() {
        free(arr_ptr as *mut u8);
        return 0;
    }

    // Copy shape values as i64
    let data_i64 = data_ptr as *mut i64;
    for (i, &val) in shape_slice[..ndim].iter().enumerate() {
        *data_i64.add(i) = val as i64;
    }

    // Fill HaxeArray fields: ptr, len, cap, elem_size
    *arr_ptr.add(0) = data_ptr as usize; // ptr
    *arr_ptr.add(1) = ndim; // len
    *arr_ptr.add(2) = cap; // cap
    *arr_ptr.add(3) = elem_size; // elem_size

    arr_ptr as i64
}

/// tensor.shape_ptr() -> i64 (returns raw pointer to shape data, for internal use)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape_ptr(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.shape as i64
}

/// tensor.shape_ndim() -> i64 (helper: returns ndim for shape access)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape_ndim(tensor_ptr: i64) -> i64 {
    rayzor_tensor_ndim(tensor_ptr)
}

// ============================================================================
// Element access
// ============================================================================

/// Flat-indexed scalar read — `tensor.getFlat(i)` reads element `i`
/// of a contiguous tensor without going through the `Array<Int>`
/// indexing path. Falls back to row-major + strides for non-contiguous
/// tensors. Eliminates the per-call Haxe array allocation that was the
/// dominant cost when looping over a 128k logits vector in
/// LocalTempSampler.sample (see profile from session 2026-06-04).
///
/// Returns 0.0 if `i` is out of range or the tensor handle is null.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get_flat(tensor_ptr: i64, i: i64) -> f64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_GET_FLAT);
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if i < 0 || (i as usize) >= t.numel {
        return 0.0;
    }
    let idx = i as usize;
    // Contiguous fast path: just data[idx * elem_size]. This is the
    // common case for the logits vector that comes out of the
    // final matmul and stays contiguous through the sampler.
    if t.owns_data {
        return load_f32_at(t.data, idx, t.dtype) as f64;
    }
    // Strided fallback: walk the strides to convert flat -> N-D offset.
    let shape_slice = std::slice::from_raw_parts(t.shape, t.ndim);
    let strides_slice = std::slice::from_raw_parts(t.strides, t.ndim);
    let mut remaining = idx;
    let mut elem_offset: usize = 0;
    for axis in (0..t.ndim).rev() {
        let dim = shape_slice[axis];
        let i_axis = remaining % dim;
        remaining /= dim;
        elem_offset += i_axis * strides_slice[axis];
    }
    load_f32_at(t.data, elem_offset, t.dtype) as f64
}

/// Flat-indexed scalar write — the store counterpart to
/// `rayzor_tensor_get_flat`. Narrows `value` to the tensor's element type
/// (`store_f32_at` dispatches on `dtype`), so writing an f64 into an F32
/// tensor stores 4 bytes. A raw `Ptr<Float>` write from Haxe would instead
/// store 8 bytes at an 8-byte stride and corrupt the buffer. No-op if `i` is
/// out of range or the handle is null.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_set_flat(tensor_ptr: i64, i: i64, value: f64) {
    if tensor_ptr == 0 {
        return;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if i < 0 || (i as usize) >= t.numel {
        return;
    }
    let idx = i as usize;
    if t.owns_data {
        store_f32_at(t.data, idx, t.dtype, value as f32);
        return;
    }
    let shape_slice = std::slice::from_raw_parts(t.shape, t.ndim);
    let strides_slice = std::slice::from_raw_parts(t.strides, t.ndim);
    let mut remaining = idx;
    let mut elem_offset: usize = 0;
    for axis in (0..t.ndim).rev() {
        let dim = shape_slice[axis];
        let i_axis = remaining % dim;
        remaining /= dim;
        elem_offset += i_axis * strides_slice[axis];
    }
    store_f32_at(t.data, elem_offset, t.dtype, value as f32);
}

/// Top-K + repetition-penalty scan in a single FFI call.
///
/// Replaces the per-element `tensor.getFlat(i)` + `adjusted(...)` loop that
/// LocalTempSampler.sample runs over a 128k-entry logits vector. The per-call
/// overhead of the extern dispatch (call instruction, parameter shuffle,
/// return) was a measurable floor on the sampler's wall — roughly 5–10 ns
/// per element × 128k elements per token = 0.6–1.3 ms/token of pure FFI
/// noise on Llama-3.2-1B. After this primitive the sampler does ONE FFI
/// call and the inner scan runs as a tight Rust loop with no boundary
/// crossings.
///
/// Semantics (byte-identical to the Haxe `LocalTempSampler.sample` scan):
/// - Walk every element `i` in the logits tensor.
/// - If `recent_ids` contains `i` AND `repetition_penalty > 1.0`:
///   `lg = (lg > 0.0) ? lg / penalty : lg * penalty`
/// - Insertion-sort the running result into `out_logits` / `out_ids`
///   descending; cutoff at `k` survivors.
/// - Returns the number of survivors actually written (≤ `k`).
///
/// Inputs:
///   - `logits_ptr` — `Tensor*` (must be F32 + owns_data; strided/non-F32
///     fall back to caller's old scan via the `-1` failure return)
///   - `out_logits_ptr` — `*mut f64` sized to at least `k`
///   - `out_ids_ptr`    — `*mut i64` sized to at least `k`
///   - `k`              — clamped to `[0, numel]`
///   - `recent_ids_ptr` — `*const i64` (may be 0/null to disable penalty)
///   - `recent_len`     — number of entries in `recent_ids_ptr`
///   - `repetition_penalty` — > 1.0 enables penalty; ≤ 1.0 is a no-op
///
/// Returns:
///   - `>= 0` — number of survivors written
///   - `-1`   — error (null tensor, non-F32 dtype, non-contiguous, ...)
///
/// SAFETY: all pointers must point to valid, sized buffers for the
/// duration of the call. The output buffers are written sequentially;
/// the caller is responsible for not aliasing them with the input.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_topk_scan(
    logits_ptr: i64,
    out_logits_ptr: i64,
    out_ids_ptr: i64,
    k: i64,
    recent_ids_ptr: i64,
    recent_len: i64,
    repetition_penalty: f64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TOPK_SCAN);
    if logits_ptr == 0 || out_logits_ptr == 0 || out_ids_ptr == 0 {
        return -1;
    }
    let t = &*(logits_ptr as *const RayzorTensor);
    // Gate the fast path on F32 + canonical row-major strides — NOT on
    // `owns_data`. The sampler's input is `lastRow(logits)`, which goes
    // through `slice` then `reshape`; both return contiguous VIEWS of the
    // lm_head output's storage with `owns_data = false`. Checking
    // `owns_data` here would bail out every time and route every sample
    // through the per-element fallback (defeating the whole point of the
    // primitive). The is_contiguous() check on strides catches the real
    // failure case (a `permute` or strided slice in some future logits
    // backend) without false-positiving on views of contiguous storage.
    if t.dtype != DTYPE_F32 || !t.is_contiguous() {
        return -1;
    }

    let n = t.numel;
    let k = (k.max(0) as usize).min(n);
    if k == 0 {
        return 0;
    }

    let src = t.data as *const f32;
    let out_logits = out_logits_ptr as *mut f64;
    let out_ids = out_ids_ptr as *mut i64;

    let penalize = repetition_penalty > 1.0 && recent_ids_ptr != 0 && recent_len > 0;
    let rp = repetition_penalty;
    let recent = if penalize {
        Some(std::slice::from_raw_parts(
            recent_ids_ptr as *const i64,
            recent_len as usize,
        ))
    } else {
        None
    };

    // Insert a candidate (lg, idx) into the top-K buffer. Caller has
    // already filtered against the cutoff in the steady-state branch.
    #[inline(always)]
    unsafe fn insert_candidate(
        lg: f64,
        idx: i64,
        out_logits: *mut f64,
        out_ids: *mut i64,
        end: usize,
    ) {
        let mut pos = end;
        while pos > 0 && *out_logits.add(pos - 1) < lg {
            *out_logits.add(pos) = *out_logits.add(pos - 1);
            *out_ids.add(pos) = *out_ids.add(pos - 1);
            pos -= 1;
        }
        *out_logits.add(pos) = lg;
        *out_ids.add(pos) = idx;
    }

    // Fill phase: insertion-sort the first k candidates so the cutoff
    // (out_logits[k-1]) is well-defined before the steady-state loop.
    let mut sz: usize = 0;
    let fill_end = k.min(n);
    for i in 0..fill_end {
        let mut lg = (*src.add(i)) as f64;
        if let Some(recent) = recent {
            if recent_contains(recent, i as i64) {
                lg = if lg > 0.0 { lg / rp } else { lg * rp };
            }
        }
        insert_candidate(lg, i as i64, out_logits, out_ids, sz);
        sz += 1;
    }

    if sz < k {
        // n < k: tiny logits buffer; nothing more to do.
        return sz as i64;
    }

    // Steady-state loop. The cutoff = out_logits[k-1] is the lowest of
    // the current top-K survivors. In typical decode (k=50, n=128k) about
    // 0.5% of logits beat it; the rest are pure fast-reject and the
    // NEON pre-filter discards them four at a time.
    let mut i = fill_end;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        while i + 4 <= n {
            let cutoff = *out_logits.add(k - 1);
            // Load 4 f32 logits, widen to 2× f64.
            let lg32 = vld1q_f32(src.add(i));
            let lg_lo = vcvt_f64_f32(vget_low_f32(lg32));
            let lg_hi = vcvt_high_f64_f32(lg32);
            // Apply repetition penalty when needed. The penalty branches
            // on a per-lane `is in recent` lookup, which is hard to
            // SIMD-fuse with the cutoff compare — fall back to scalar
            // for the penalize path's pre-filter.
            if penalize {
                // Scalar fast path: still amortise the load by computing
                // the four f64s in one vector pair and storing to a
                // tiny stack buffer.
                let mut buf = [0f64; 4];
                vst1q_f64(buf.as_mut_ptr(), lg_lo);
                vst1q_f64(buf.as_mut_ptr().add(2), lg_hi);
                let recent = recent.unwrap_unchecked();
                for (j, &raw) in buf.iter().enumerate() {
                    let mut lg = raw;
                    if recent_contains(recent, (i + j) as i64) {
                        lg = if lg > 0.0 { lg / rp } else { lg * rp };
                    }
                    if lg > cutoff {
                        insert_candidate(lg, (i + j) as i64, out_logits, out_ids, k - 1);
                    }
                }
            } else {
                // No penalty: SIMD pre-filter against the cutoff.
                let cutoff_v = vdupq_n_f64(cutoff);
                let mask_lo = vcgtq_f64(lg_lo, cutoff_v);
                let mask_hi = vcgtq_f64(lg_hi, cutoff_v);
                let any_passes =
                    vmaxvq_u32(vreinterpretq_u32_u64(vorrq_u64(mask_lo, mask_hi))) != 0;
                if any_passes {
                    let mut buf = [0f64; 4];
                    vst1q_f64(buf.as_mut_ptr(), lg_lo);
                    vst1q_f64(buf.as_mut_ptr().add(2), lg_hi);
                    for (j, &lg) in buf.iter().enumerate() {
                        // Re-check against the latest cutoff — earlier
                        // lanes in this same chunk may have raised it.
                        if lg > *out_logits.add(k - 1) {
                            insert_candidate(lg, (i + j) as i64, out_logits, out_ids, k - 1);
                        }
                    }
                }
            }
            i += 4;
        }
    }
    // Scalar tail (and the path taken on non-aarch64).
    while i < n {
        let mut lg = (*src.add(i)) as f64;
        if let Some(recent) = recent {
            if recent_contains(recent, i as i64) {
                lg = if lg > 0.0 { lg / rp } else { lg * rp };
            }
        }
        if lg > *out_logits.add(k - 1) {
            insert_candidate(lg, i as i64, out_logits, out_ids, k - 1);
        }
        i += 1;
    }

    sz as i64
}

/// tensor.get(indices_ptr, ndim) -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get(tensor_ptr: i64, indices_ptr: i64, ndim: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    let indices = read_shape(indices_ptr, ndim as usize);
    let off = t.offset(&indices);
    load_f32_at(t.data, off, t.dtype) as f64
}

/// tensor.set(indices_ptr, ndim, value) -> void
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_set(
    tensor_ptr: i64,
    indices_ptr: i64,
    ndim: i64,
    value: f64,
) {
    if tensor_ptr == 0 {
        return;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    let indices = read_shape(indices_ptr, ndim as usize);
    let off = t.offset(&indices);
    store_f32_at(t.data, off, t.dtype, value as f32);
}

/// Bulk copy `src.shape[0]` contiguous rows from `src` into `dst` starting at
/// row index `dst_row_offset` along axis 0. Both tensors must be F32 and must
/// share the same trailing-axis sizes (`shape[1..]`). Returns 0 on success,
/// -1 on null pointer, dtype mismatch, shape mismatch, or out-of-bounds.
///
/// This is the bulk-row sibling of `rayzor_tensor_set`, intended for code that
/// concatenates / appends row-blocks (KV-cache appends, sequence-dim grows).
/// Falls back to scalar set semantics conceptually but skips the index-walk
/// and the per-element `store_f32_at` dispatch, so it's ~headroom faster on
/// large blocks while still being a single memcpy.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_append_along_0_f32(
    dst_ptr: i64,
    src_ptr: i64,
    dst_row_offset: i64,
) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_append_along_0_f32");
    if dst_ptr == 0 || src_ptr == 0 {
        return -1;
    }
    let dst = &*(dst_ptr as *const RayzorTensor);
    let src = &*(src_ptr as *const RayzorTensor);

    if dst.dtype != DTYPE_F32 || src.dtype != DTYPE_F32 {
        return -1;
    }
    if dst.ndim == 0 || src.ndim == 0 || dst.ndim != src.ndim {
        return -1;
    }

    let dst_shape = std::slice::from_raw_parts(dst.shape, dst.ndim);
    let src_shape = std::slice::from_raw_parts(src.shape, src.ndim);

    // Trailing axes must match (shape[1..]) so the row layout is identical.
    for i in 1..dst.ndim {
        if dst_shape[i] != src_shape[i] {
            return -1;
        }
    }

    // Row stride in elements = product of shape[1..]. Equals dst.strides[0]
    // for contiguous f32, but compute from shape so this is safe regardless.
    let row_stride_elements: usize = dst_shape[1..].iter().product();
    let n_rows_to_copy = src_shape[0];

    if dst_row_offset < 0 {
        return -1;
    }
    let dst_row_off = dst_row_offset as usize;
    if dst_row_off + n_rows_to_copy > dst_shape[0] {
        return -1;
    }

    let byte_count = n_rows_to_copy * row_stride_elements * 4;
    let dst_offset_bytes = dst_row_off * row_stride_elements * 4;

    std::ptr::copy_nonoverlapping(src.data, dst.data.add(dst_offset_bytes), byte_count);
    0
}

/// Broadcast `src` along axis 0 by repeating each row `repeats` times,
/// writing into `dst`. Both tensors must be F32 and must share trailing-axis
/// sizes (`shape[1..]`). `dst.shape[0]` must be at least
/// `src.shape[0] * repeats`. Returns 0 on success, -1 on validation failure.
///
/// Layout: src row i is written to dst rows `i*repeats .. i*repeats+repeats`,
/// which matches numpy's `np.repeat(x, repeats, axis=0)` (KV-head GQA expand
/// convention), not `np.tile`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_broadcast_repeat_0_f32(
    dst_ptr: i64,
    src_ptr: i64,
    repeats: i64,
) -> i64 {
    if dst_ptr == 0 || src_ptr == 0 {
        return -1;
    }
    let dst = &*(dst_ptr as *const RayzorTensor);
    let src = &*(src_ptr as *const RayzorTensor);

    if dst.dtype != DTYPE_F32 || src.dtype != DTYPE_F32 {
        return -1;
    }
    if dst.ndim == 0 || src.ndim == 0 || dst.ndim != src.ndim {
        return -1;
    }
    if repeats <= 0 {
        return -1;
    }

    let dst_shape = std::slice::from_raw_parts(dst.shape, dst.ndim);
    let src_shape = std::slice::from_raw_parts(src.shape, src.ndim);

    for i in 1..dst.ndim {
        if dst_shape[i] != src_shape[i] {
            return -1;
        }
    }

    let repeats = repeats as usize;
    if src_shape[0].saturating_mul(repeats) > dst_shape[0] {
        return -1;
    }

    let row_size_elements: usize = src_shape[1..].iter().product();
    let row_size_bytes = row_size_elements * 4;

    for i in 0..src_shape[0] {
        let src_row = src.data.add(i * row_size_bytes);
        for r in 0..repeats {
            let dst_row_idx = i * repeats + r;
            let dst_row = dst.data.add(dst_row_idx * row_size_bytes);
            std::ptr::copy_nonoverlapping(src_row, dst_row, row_size_bytes);
        }
    }
    0
}

/// GQA KV-head expansion. Source `src` has shape `[seqK, num_kv_heads, head_dim]`
/// (KV-heads on axis 1, as produced by KVCache views). Output has shape
/// `[num_kv_heads * repeats, seqK, head_dim]` with the axis-0/axis-1 swap
/// baked in, such that `out[qh, j, d] = src[j, qh / repeats, d]`. Equivalent
/// to `src.permute([1,0,2]).broadcastRepeat0(repeats)` but a single strided
/// memcpy per `(qh, j)` pair instead of an O(qh * j * d) element-walk.
///
/// Allocates and returns a fresh F32 tensor; returns 0 on null pointer, dtype
/// mismatch (F32 only), shape mismatch (ndim != 3), or `repeats <= 0`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_expand_kv_heads_axis1_f32(
    src_ptr: i64,
    repeats: i64,
) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_expand_kv_heads_axis1_f32");
    if src_ptr == 0 || repeats <= 0 {
        return 0;
    }
    let src = &*(src_ptr as *const RayzorTensor);
    if src.dtype != DTYPE_F32 || src.ndim != 3 {
        return 0;
    }

    let src_shape = std::slice::from_raw_parts(src.shape, 3);
    let src_strides = std::slice::from_raw_parts(src.strides, 3);
    let seq_k = src_shape[0];
    let num_kv_heads = src_shape[1];
    let head_dim = src_shape[2];
    let repeats = repeats as usize;
    let num_q_heads = num_kv_heads * repeats;

    // Innermost dim must be contiguous so each (qh, j) write is a single
    // memcpy. If src was produced by a non-contiguous view (permute /
    // transposeLast2), fall through to the scalar Haxe path by returning 0.
    if src_strides[2] != 1 {
        return 0;
    }

    let out_shape = [num_q_heads, seq_k, head_dim];
    let result = alloc_tensor(&out_shape, DTYPE_F32, Some(0.0));
    if result == 0 {
        return 0;
    }
    let dst = &*(result as *const RayzorTensor);

    let src_stride_j = src_strides[0]; // elements between j and j+1
    let src_stride_kvh = src_strides[1]; // elements between kvh and kvh+1
                                         // Output is freshly allocated contiguous row-major:
                                         //   dst[qh, j, d] at offset qh*seq_k*head_dim + j*head_dim + d
    let row_bytes = head_dim * 4;
    let dst_row_stride_elements = seq_k * head_dim;

    for qh in 0..num_q_heads {
        let kvh = qh / repeats;
        let dst_head_off = qh * dst_row_stride_elements;
        let src_head_off = kvh * src_stride_kvh;
        for j in 0..seq_k {
            let src_off = src_head_off + j * src_stride_j;
            let dst_off = dst_head_off + j * head_dim;
            std::ptr::copy_nonoverlapping(
                src.data.add(src_off * 4),
                dst.data.add(dst_off * 4),
                row_bytes,
            );
        }
    }

    result
}

// ============================================================================
// Reshape / Transpose
// ============================================================================

/// tensor.reshape(shape_ptr, ndim) -> i64 (new tensor, shared data)
#[no_mangle]
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
pub unsafe extern "C" fn rayzor_tensor_reshape(tensor_ptr: i64, shape_ptr: i64, ndim: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_RESHAPE);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_reshape");
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    let new_shape = read_shape(shape_ptr, ndim as usize);
    let new_numel: usize = new_shape.iter().product();

    // Verify numel matches
    if new_numel != t.numel {
        return 0; // shape mismatch
    }

    let new_ndim = new_shape.len();

    // numpy/torch semantics: reshape only returns a view when the source
    // memory is already laid out in the requested order — i.e. the source
    // is contiguous in its CURRENT shape. After `permute([1, 0, 2])` the
    // strides are non-canonical and the data isn't laid out in the new
    // shape's order, so a view would mean every subsequent read using the
    // freshly-computed contiguous strides lands on the wrong element.
    // For the GQAttention out-projection that meant garbage hidden states
    // (`context.permute([1,0,2]).reshape([seqQ, numQHeads*headDim])`),
    // which is one of the dominant remaining coherence bugs.
    //
    // Detect non-contiguous sources and materialise: walk the source via
    // its real strides into a fresh contiguous buffer, then return a
    // contiguous tensor with the new shape.
    let src_shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let src_strides = std::slice::from_raw_parts(t.strides, t.ndim);
    let canonical_strides = RayzorTensor::compute_strides(src_shape);
    let is_contig = src_strides == canonical_strides.as_slice();

    if is_contig {
        // Allocate new shape
        let new_shape_ptr = malloc(new_ndim * std::mem::size_of::<usize>()) as *mut usize;
        if new_shape_ptr.is_null() {
            return 0;
        }
        for i in 0..new_ndim {
            *new_shape_ptr.add(i) = new_shape[i];
        }

        // Compute new strides
        let strides = RayzorTensor::compute_strides(&new_shape);
        let new_strides_ptr = malloc(new_ndim * std::mem::size_of::<usize>()) as *mut usize;
        if new_strides_ptr.is_null() {
            free(new_shape_ptr as *mut u8);
            return 0;
        }
        for i in 0..new_ndim {
            *new_strides_ptr.add(i) = strides[i];
        }

        // Allocate new tensor struct (view — shares data)
        let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
        if new_t.is_null() {
            free(new_shape_ptr as *mut u8);
            free(new_strides_ptr as *mut u8);
            return 0;
        }

        *new_t = RayzorTensor {
            data: t.data, // shared
            shape: new_shape_ptr,
            strides: new_strides_ptr,
            ndim: new_ndim,
            numel: new_numel,
            dtype: t.dtype,
            owns_data: false, // view
            device: t.device,
            numa_node: t.numa_node,
            refcount: std::sync::atomic::AtomicUsize::new(1),
            parent: tensor_ptr as *mut RayzorTensor,
        };
        // View bumps parent's refcount so the parent's data buffer stays
        // alive until every view of it is also freed.
        t.refcount
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        return new_t as i64;
    }

    // Materialise the strided source into a fresh contiguous tensor with
    // the new shape. Walk the source's element-by-element using its real
    // strides; write into the new tensor's row-major linear order.
    let result = alloc_tensor(&new_shape, t.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);

    // Multi-dim iteration via index vector over the SOURCE shape; the
    // linear write index into the new contiguous buffer increments
    // monotonically because both source and dest visit the same numel.
    let src_ndim = t.ndim;
    let mut idx = vec![0usize; src_ndim];
    for linear in 0..t.numel {
        // Compute source memory offset from current multi-index + strides.
        let mut src_off = 0usize;
        for (axis, &i) in idx.iter().enumerate() {
            src_off += i * src_strides[axis];
        }
        let v = load_f32_at(t.data, src_off, t.dtype);
        store_f32_at(r.data, linear, t.dtype, v);
        // Increment multi-index (rightmost-axis varies fastest).
        for axis in (0..src_ndim).rev() {
            idx[axis] += 1;
            if idx[axis] < src_shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }

    result
}

/// tensor.permute(axes_ptr, ndim) -> i64 (n-D permutation — reorders shape/strides, view)
#[no_mangle]
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
pub unsafe extern "C" fn rayzor_tensor_permute(
    tensor_ptr: i64,
    axes_ptr: i64,
    axes_len: i64,
) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_permute");
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    let n = axes_len as usize;
    if n != t.ndim {
        return 0;
    }

    let axes_data = axes_ptr as *const i64;
    let mut seen = vec![false; n];
    let mut axes = vec![0usize; n];
    for i in 0..n {
        let ax = *axes_data.add(i) as usize;
        if ax >= n || seen[ax] {
            return 0;
        }
        seen[ax] = true;
        axes[i] = ax;
    }

    let old_shape = std::slice::from_raw_parts(t.shape, n);
    let old_strides = std::slice::from_raw_parts(t.strides, n);

    let new_shape_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }
    for i in 0..n {
        *new_shape_ptr.add(i) = old_shape[axes[i]];
        *new_strides_ptr.add(i) = old_strides[axes[i]];
    }

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        return 0;
    }
    *new_t = RayzorTensor {
        data: t.data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: n,
        numel: t.numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: tensor_ptr as *mut RayzorTensor,
    };
    t.refcount
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    new_t as i64
}

/// tensor.slice(dim, start, end) -> i64 (view over [start..end) along `dim`, view)
#[no_mangle]
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
pub unsafe extern "C" fn rayzor_tensor_slice(
    tensor_ptr: i64,
    dim: i64,
    start: i64,
    end: i64,
) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_slice");
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    let d = dim as usize;
    if d >= t.ndim {
        return 0;
    }

    let old_shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let old_strides = std::slice::from_raw_parts(t.strides, t.ndim);
    let dim_size = old_shape[d];

    let s = start.max(0) as usize;
    let e = (end as usize).min(dim_size);
    if s >= e {
        return 0;
    }
    let new_dim_size = e - s;

    let new_shape_ptr = malloc(t.ndim * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(t.ndim * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }
    for i in 0..t.ndim {
        *new_shape_ptr.add(i) = if i == d { new_dim_size } else { old_shape[i] };
        *new_strides_ptr.add(i) = old_strides[i];
    }

    let mut new_numel = 1usize;
    for i in 0..t.ndim {
        new_numel *= *new_shape_ptr.add(i);
    }

    // Offset data pointer by s * stride[d] elements
    let elem_size = dtype_size(t.dtype);
    let byte_offset = s * old_strides[d] * elem_size;
    let new_data = t.data.add(byte_offset);

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        return 0;
    }
    *new_t = RayzorTensor {
        data: new_data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: t.ndim,
        numel: new_numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: tensor_ptr as *mut RayzorTensor,
    };
    t.refcount
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    new_t as i64
}

/// tensor.transpose() -> i64 (2D transpose — swaps shape/strides)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_transpose(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    if t.ndim != 2 {
        return tensor_ptr;
    } // no-op for non-2D

    let old_shape = std::slice::from_raw_parts(t.shape, 2);
    let old_strides = std::slice::from_raw_parts(t.strides, 2);

    let new_shape_ptr = malloc(2 * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(2 * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }

    *new_shape_ptr.add(0) = old_shape[1];
    *new_shape_ptr.add(1) = old_shape[0];
    *new_strides_ptr.add(0) = old_strides[1];
    *new_strides_ptr.add(1) = old_strides[0];

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        return 0;
    }

    *new_t = RayzorTensor {
        data: t.data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: 2,
        numel: t.numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: tensor_ptr as *mut RayzorTensor,
    };
    t.refcount
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    new_t as i64
}

// ============================================================================
// Elementwise arithmetic
// ============================================================================

/// Set up output tensor and return (a_slice, b_slice, r_slice) for an
/// elementwise binary f32 op. Returns 0 result and `None` if the op is
/// invalid (shape mismatch, non-f32 dtype). Assumes contiguous f32 data.
#[allow(clippy::type_complexity)]
unsafe fn prepare_binop<'a>(
    a_ptr: i64,
    b_ptr: i64,
) -> Option<(&'a [f32], &'a [f32], &'a mut [f32], i64)> {
    if a_ptr == 0 || b_ptr == 0 {
        return None;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    if a.numel != b.numel || a.dtype != DTYPE_F32 || b.dtype != DTYPE_F32 {
        return None;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return None;
    }

    let r = &*(result as *const RayzorTensor);
    let n = a.numel;
    let a_slice = std::slice::from_raw_parts(a.data as *const f32, n);
    let b_slice = std::slice::from_raw_parts(b.data as *const f32, n);
    let r_slice = std::slice::from_raw_parts_mut(r.data as *mut f32, n);
    Some((a_slice, b_slice, r_slice, result))
}

/// Row-broadcast f32 binop: `a [..., D] op b [D]` → result with `a`'s shape.
///
/// Common LLM pattern: RMSNorm/LayerNorm gain, dense-layer bias. Walks
/// `a` in `last`-sized groups and applies the kernel against `b` slice
/// repeated for each group. Returns 0 if the shapes don't match this
/// exact "trailing-dim broadcast" form — the caller falls through to
/// the elementwise scalar path which also fails on shape mismatch.
unsafe fn tensor_binop_row_broadcast(
    a_ptr: i64,
    b_ptr: i64,
    kernel: fn(&mut [f32], &[f32], &[f32]),
) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.dtype != DTYPE_F32 || b.dtype != DTYPE_F32 {
        return 0;
    }
    if a.ndim == 0 || b.ndim != 1 {
        return 0;
    }
    let a_shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let b_shape = std::slice::from_raw_parts(b.shape, 1);
    let last = a_shape[a.ndim - 1];
    if b_shape[0] != last || !a.numel.is_multiple_of(last) {
        return 0;
    }

    let result = alloc_tensor(a_shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let b_data = b.data as *const f32;
    let r_data = r.data as *mut f32;
    let b_slice = std::slice::from_raw_parts(b_data, last);
    let groups = a.numel / last;
    for g in 0..groups {
        let off = g * last;
        let a_row = std::slice::from_raw_parts(a_data.add(off), last);
        let r_row = std::slice::from_raw_parts_mut(r_data.add(off), last);
        kernel(r_row, a_row, b_slice);
    }
    result
}

/// Scalar fallback for elementwise binary ops on non-f32 dtypes.
/// Both inputs must share dtype + numel. The output tensor is allocated
/// in the same dtype, kernel runs in f32 in-register.
unsafe fn tensor_binop_scalar(a_ptr: i64, b_ptr: i64, op: fn(f32, f32) -> f32) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.numel != b.numel || a.dtype != b.dtype {
        return 0;
    }
    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    for i in 0..a.numel {
        let av = load_f32_at(a.data, i, a.dtype);
        let bv = load_f32_at(b.data, i, a.dtype);
        store_f32_at(r.data, i, a.dtype, op(av, bv));
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_add(a: i64, b: i64) -> i64 {
    if let Some((a_s, b_s, r_s, result)) = prepare_binop(a, b) {
        crate::tensor_simd::add_slice(r_s, a_s, b_s);
        return result;
    }
    let broadcast = tensor_binop_row_broadcast(a, b, crate::tensor_simd::add_slice);
    if broadcast != 0 {
        return broadcast;
    }
    tensor_binop_scalar(a, b, |x, y| x + y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sub(a: i64, b: i64) -> i64 {
    if let Some((a_s, b_s, r_s, result)) = prepare_binop(a, b) {
        crate::tensor_simd::sub_slice(r_s, a_s, b_s);
        return result;
    }
    let broadcast = tensor_binop_row_broadcast(a, b, crate::tensor_simd::sub_slice);
    if broadcast != 0 {
        return broadcast;
    }
    tensor_binop_scalar(a, b, |x, y| x - y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mul(a: i64, b: i64) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_mul");
    if let Some((a_s, b_s, r_s, result)) = prepare_binop(a, b) {
        crate::tensor_simd::mul_slice(r_s, a_s, b_s);
        return result;
    }
    let broadcast = tensor_binop_row_broadcast(a, b, crate::tensor_simd::mul_slice);
    if broadcast != 0 {
        return broadcast;
    }
    tensor_binop_scalar(a, b, |x, y| x * y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_silu_mul(a: i64, b: i64) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_silu_mul");
    if let Some((a_s, b_s, r_s, result)) = prepare_binop(a, b) {
        let n = a_s.len();
        let threads = crate::worker_pool::auto_kernel_threads();
        let threshold = crate::env_var(
            "RZT_SILU_MUL_PAR_THRESHOLD",
            "RAYZOR_SILU_MUL_PAR_THRESHOLD",
        )
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(65_536);
        if threads > 1 && n >= threshold {
            let a_addr = a_s.as_ptr() as usize;
            let b_addr = b_s.as_ptr() as usize;
            let r_addr = r_s.as_mut_ptr() as usize;
            crate::worker_pool::global().parallel_rows(n, threads, move |lo, hi| unsafe {
                let a_ptr = a_addr as *const f32;
                let b_ptr = b_addr as *const f32;
                let r_ptr = r_addr as *mut f32;
                for i in lo..hi {
                    let x = *a_ptr.add(i);
                    *r_ptr.add(i) = (x / (1.0 + (-x).exp())) * *b_ptr.add(i);
                }
            });
        } else {
            for i in 0..n {
                let x = a_s[i];
                r_s[i] = (x / (1.0 + (-x).exp())) * b_s[i];
            }
        }
        return result;
    }
    tensor_binop_scalar(a, b, |x, y| (x / (1.0 + (-x).exp())) * y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_div(a: i64, b: i64) -> i64 {
    match prepare_binop(a, b) {
        Some((a_s, b_s, r_s, result)) => {
            crate::tensor_simd::div_slice(r_s, a_s, b_s);
            result
        }
        None => tensor_binop_scalar(a, b, |x, y| x / y),
    }
}

/// In-place elementwise add: `dest[i] += src[i]`. Returns `dest` unchanged so
/// the call site can rebind (`x = x.addInto(y)`) without allocating a fresh
/// result tensor. Saves one [..., hidden] F32 alloc per residual add in the
/// transformer hot loop (TransformerBlock attention + FFN residuals are the
/// main consumers).
///
/// Safety invariants enforced (panics with a `eprintln!` + `std::process::abort`
/// on violation — these are programmer errors, not recoverable conditions):
///   (a) Both tensors non-null with matching `ndim`, `numel`, and shape.
///   (b) Matching dtype.
///   (c) `dest` is contiguous in its current shape. `src` strides are
///       accommodated on the slow path; a non-contiguous `dest` would either
///       silently skip elements (fast path) or accumulate into the wrong slot
///       (strided path) so we reject it outright.
///   (d) `dest.data != src.data` — aliasing would double-count on the SIMD
///       path. Not a current call site but cheap insurance.
///
/// F32 fast path: when both tensors are contiguous F32, dispatches to
/// `tensor_simd::add_slice(dst, dst, src)` (NEON vaddq_f32 4-lane on
/// aarch64, SSE2 _mm_add_ps on x86_64, scalar elsewhere). `add_slice`
/// computes `r[i] = a[i] + b[i]`, which is exactly the in-place form when
/// the result and first-operand slices alias the same buffer — verified to
/// be safe because both SIMD backends load a, load b, op, then store before
/// advancing to the next chunk.
///
/// F16 / BF16: not currently exercised by the transformer residual paths
/// (those tensors are F32 throughout in Llama / GPT-style models). Falls
/// back to a strided scalar loop via `load_f32_at` / `store_f32_at` so the
/// call doesn't silently no-op, but emits a one-line `eprintln!` on the
/// first hit to flag the slow path. Other dtypes (I32, I8, U8, FP8) abort.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_add_into(dest: i64, src: i64) {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_ADD_INTO);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_add_into");
    if dest == 0 || src == 0 {
        eprintln!(
            "rayzor_tensor_add_into: null tensor pointer (dest={:#x}, src={:#x})",
            dest, src
        );
        std::process::abort();
    }
    let d = &*(dest as *const RayzorTensor);
    let s = &*(src as *const RayzorTensor);

    // Under @:shared (Arc-backed Tensor), strict-move's compile-time
    // single-owner guarantee is replaced by runtime refcounting. addInto
    // mutates the receiver in place, so two aliased bindings to the
    // same Arc would BOTH observe the mutation (silent UAF-class bug).
    // Debug-builds trap on shared dest; release builds rely on caller
    // discipline. Current nue/* call sites (TransformerBlock, LayerNorm,
    // Linear) audit safe — receiver is always a freshly-produced kernel
    // output with refcount==1. Trap exists to catch the next user-code
    // call site that breaks the convention. Cost: one Acquire atomic
    // load per addInto call (~5ns vs the kernel's microseconds).
    debug_assert!(
        d.refcount.load(std::sync::atomic::Ordering::Acquire) == 1,
        "rayzor_tensor_add_into: dest has shared refcount > 1; in-place mutation would silently leak to aliased bindings. Use addInto only on uniquely-owned tensors (freshly produced or after deepClone)."
    );

    // (a) shape compatibility
    if d.ndim != s.ndim || d.numel != s.numel {
        eprintln!(
            "rayzor_tensor_add_into: shape mismatch — dest.ndim={}, src.ndim={}, dest.numel={}, src.numel={}",
            d.ndim, s.ndim, d.numel, s.numel
        );
        std::process::abort();
    }
    let d_shape = std::slice::from_raw_parts(d.shape, d.ndim);
    let s_shape = std::slice::from_raw_parts(s.shape, s.ndim);
    for i in 0..d.ndim {
        if d_shape[i] != s_shape[i] {
            eprintln!(
                "rayzor_tensor_add_into: shape mismatch at dim {} — dest={:?}, src={:?}",
                i, d_shape, s_shape
            );
            std::process::abort();
        }
    }

    // (b) dtype compatibility
    if d.dtype != s.dtype {
        eprintln!(
            "rayzor_tensor_add_into: dtype mismatch — dest.dtype={}, src.dtype={}",
            d.dtype, s.dtype
        );
        std::process::abort();
    }

    // (c) dest contiguity
    if !d.is_contiguous() {
        let d_strides = std::slice::from_raw_parts(d.strides, d.ndim);
        eprintln!(
            "rayzor_tensor_add_into: dest must be contiguous — shape={:?}, strides={:?}",
            d_shape, d_strides
        );
        std::process::abort();
    }

    // (d) aliasing — same backing buffer would double-count on SIMD
    if d.data == s.data {
        eprintln!(
            "rayzor_tensor_add_into: dest and src share the same backing buffer ({:?}); aliasing would double-count",
            d.data
        );
        std::process::abort();
    }

    // numel == 0 is a no-op; alloc_tensor allocates a 1-byte sentinel for
    // empty tensors so the pointer is non-null but there's nothing to add.
    if d.numel == 0 {
        return;
    }

    match d.dtype {
        DTYPE_F32 => {
            let n = d.numel;
            let dst_slice = std::slice::from_raw_parts_mut(d.data as *mut f32, n);
            if s.is_contiguous() {
                // Fast path: both contiguous F32. `add_assign_slice` takes a
                // single `&mut [f32]` + `&[f32]` pair, so there is no aliased
                // mutable+immutable reference to the dst memory — the SIMD
                // intrinsics inside operate on raw pointers derived once from
                // `dst_slice.as_mut_ptr()`.
                let src_slice = std::slice::from_raw_parts(s.data as *const f32, n);
                crate::tensor_simd::add_assign_slice(dst_slice, src_slice);
            } else {
                // Strided src gather: walk via src strides, accumulate into
                // dest's contiguous slot. Recompute the multi-index from the
                // linear contiguous counter using dest's shape (which equals
                // src's shape — verified above).
                let s_strides = std::slice::from_raw_parts(s.strides, s.ndim);
                let s_data = s.data as *const f32;
                let mut idx = vec![0usize; d.ndim];
                for (flat, dst_elem) in dst_slice.iter_mut().enumerate() {
                    // Compute multi-index in dest's row-major layout
                    let mut rem = flat;
                    for k in 0..d.ndim {
                        let stride: usize = d_shape[k + 1..].iter().product();
                        idx[k] = rem / stride;
                        rem %= stride;
                    }
                    // Apply src strides
                    let mut s_off: usize = 0;
                    for k in 0..s.ndim {
                        s_off += idx[k] * s_strides[k];
                    }
                    *dst_elem += *s_data.add(s_off);
                }
            }
        }
        DTYPE_F16 | DTYPE_BF16 => {
            eprintln!("rayzor_tensor_add_into: F16/BF16 not yet supported, falling back to scalar");
            // Scalar fallback: respects src strides via load_f32_at on the
            // gathered offset. Dest is contiguous so the linear counter
            // doubles as the dest offset.
            let s_strides = std::slice::from_raw_parts(s.strides, s.ndim);
            let mut idx = vec![0usize; d.ndim];
            for flat in 0..d.numel {
                let mut rem = flat;
                for k in 0..d.ndim {
                    let stride: usize = d_shape[k + 1..].iter().product();
                    idx[k] = rem / stride;
                    rem %= stride;
                }
                let mut s_off: usize = 0;
                for k in 0..s.ndim {
                    s_off += idx[k] * s_strides[k];
                }
                let dv = load_f32_at(d.data, flat, d.dtype);
                let sv = load_f32_at(s.data, s_off, s.dtype);
                store_f32_at(d.data, flat, d.dtype, dv + sv);
            }
        }
        _ => {
            eprintln!(
                "rayzor_tensor_add_into: unsupported dtype {} (only F32, F16, BF16 currently handled)",
                d.dtype
            );
            std::process::abort();
        }
    }
}

// ============================================================================
// Unary math ops
// ============================================================================

unsafe fn tensor_unary(a_ptr: i64, op: fn(f32) -> f32) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);

    if a.dtype == DTYPE_F32 {
        // Fast path: contiguous f32 → SIMD-friendly straight loop. The
        // SIMD-specialised unary kernels live in tensor_simd; non-SIMD
        // ops (e.g. transcendentals) stay scalar but in-register.
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for i in 0..a.numel {
            *r_data.add(i) = op(*a_data.add(i));
        }
    } else {
        // Generic dtype path — convert to f32, compute, convert back.
        for i in 0..a.numel {
            let v = load_f32_at(a.data, i, a.dtype);
            store_f32_at(r.data, i, a.dtype, op(v));
        }
    }

    result
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sqrt(a: i64) -> i64 {
    tensor_unary(a, |x| x.sqrt())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_exp(a: i64) -> i64 {
    tensor_unary(a, |x| x.exp())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_log(a: i64) -> i64 {
    tensor_unary(a, |x| x.ln())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_relu(a_ptr: i64) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let n = a.numel;

    if a.dtype == DTYPE_F32 {
        let a_s = std::slice::from_raw_parts(a.data as *const f32, n);
        let r_s = std::slice::from_raw_parts_mut(r.data as *mut f32, n);
        crate::tensor_simd::relu_slice(r_s, a_s);
    } else {
        for i in 0..n {
            let v = load_f32_at(a.data, i, a.dtype);
            store_f32_at(r.data, i, a.dtype, v.max(0.0));
        }
    }
    result
}

/// GELU (approximate, tanh-based) — matches PyTorch `gelu(approximate='tanh')`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gelu(a: i64) -> i64 {
    tensor_unary(a, |x| {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    })
}

/// SiLU / swish: x * sigmoid(x).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_silu(a: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_SILU);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_silu");
    // NEON silu (vectorized Cephes exp) exists behind RZT_NEON_SILU=1
    // but is OFF by default: decode A/B on Llama 3.2 1B lost all three
    // ABBA pairs (-3/-12/-6 tok/s under thermal drift). The 17µs/call
    // sizing that motivated it came from a KERNEL_TIMING run whose
    // per-call inflation overstated the true cost (~0.1ms/token, under
    // the noise floor), and NEON divide latency eats the exp saving at
    // ffn=8192. Re-evaluate on models with larger FFN widths. Output is
    // ~1-2 ULP off libm (canonical-prompt gate passed when tested).
    #[cfg(target_arch = "aarch64")]
    {
        if a != 0 && neon_silu_opted_in() {
            let t = &*(a as *const RayzorTensor);
            if t.dtype == DTYPE_F32 && t.is_contiguous() {
                let shape = std::slice::from_raw_parts(t.shape, t.ndim);
                let result = alloc_tensor(shape, t.dtype, None);
                if result != 0 {
                    let r = &*(result as *const RayzorTensor);
                    crate::tensor_simd::silu_slice_neon(
                        t.data as *const f32,
                        r.data as *mut f32,
                        t.numel,
                    );
                    return result;
                }
            }
        }
    }
    tensor_unary(a, |x| x / (1.0 + (-x).exp()))
}

#[cfg(target_arch = "aarch64")]
fn neon_silu_opted_in() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED
        .get_or_init(|| crate::env_var("RZT_NEON_SILU", "RAYZOR_NEON_SILU").is_ok_and(|v| v == "1"))
}

/// Softmax over the last dimension.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_softmax(a_ptr: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_SOFTMAX);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_softmax");
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for g in 0..groups {
            let base = g * last;
            let a_row = std::slice::from_raw_parts(a_data.add(base), last);
            let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
            let maxv = crate::tensor_simd::max_slice(a_row);
            for i in 0..last {
                r_row[i] = (a_row[i] - maxv).exp();
            }
            let sum = crate::tensor_simd::sum_slice(r_row);
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in r_row.iter_mut() {
                    *v *= inv;
                }
            }
        }
        return result;
    }

    // Generic dtype path: f32-in-register softmax with storage conversion.
    let mut row_buf = vec![0.0f32; last];
    for g in 0..groups {
        let base = g * last;
        for (i, slot) in row_buf.iter_mut().enumerate() {
            *slot = load_f32_at(a.data, base + i, a.dtype);
        }
        let mut maxv = f32::NEG_INFINITY;
        for &v in &row_buf {
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row_buf.iter_mut() {
            *v = (*v - maxv).exp();
            sum += *v;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for (i, &v) in row_buf.iter().enumerate() {
            store_f32_at(r.data, base + i, a.dtype, v * inv);
        }
    }
    result
}

/// Layer normalization over the last dimension. (x - mean) / sqrt(var + eps).
/// `eps` is passed as f64 from Haxe.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_layer_norm(a_ptr: i64, eps: f64) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps_f32 = eps as f32;
    let n = last as f32;

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for g in 0..groups {
            let base = g * last;
            let a_row = std::slice::from_raw_parts(a_data.add(base), last);
            let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
            let mean = crate::tensor_simd::sum_slice(a_row) / n;
            crate::tensor_simd::sub_const_slice(r_row, a_row, mean);
            let var = crate::tensor_simd::sum_of_squares(r_row) / n;
            let inv = 1.0 / (var + eps_f32).sqrt();
            for v in r_row.iter_mut() {
                *v *= inv;
            }
        }
        return result;
    }

    // Generic dtype path: f32-in-register stats with storage conversion.
    let mut row_buf = vec![0.0f32; last];
    for g in 0..groups {
        let base = g * last;
        let mut sum = 0.0f32;
        for (i, slot) in row_buf.iter_mut().enumerate() {
            *slot = load_f32_at(a.data, base + i, a.dtype);
            sum += *slot;
        }
        let mean = sum / n;
        let mut sumsq = 0.0f32;
        for v in row_buf.iter_mut() {
            *v -= mean;
            sumsq += *v * *v;
        }
        let inv = 1.0 / (sumsq / n + eps_f32).sqrt();
        for (i, &v) in row_buf.iter().enumerate() {
            store_f32_at(r.data, base + i, a.dtype, v * inv);
        }
    }
    result
}

/// RMS normalization over the last dimension. x / sqrt(mean(x^2) + eps).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rms_norm(a_ptr: i64, eps: f64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_RMS_NORM);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_rms_norm");
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps_f32 = eps as f32;
    let n = last as f32;

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for g in 0..groups {
            let base = g * last;
            let a_row = std::slice::from_raw_parts(a_data.add(base), last);
            let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
            let ms = crate::tensor_simd::sum_of_squares(a_row) / n;
            let inv = 1.0 / (ms + eps_f32).sqrt();
            crate::tensor_simd::mul_const_slice(r_row, a_row, inv);
        }
        return result;
    }

    for g in 0..groups {
        let base = g * last;
        let mut sumsq = 0.0f32;
        for i in 0..last {
            let v = load_f32_at(a.data, base + i, a.dtype);
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n + eps_f32).sqrt();
        for i in 0..last {
            let v = load_f32_at(a.data, base + i, a.dtype);
            store_f32_at(r.data, base + i, a.dtype, v * inv);
        }
    }
    result
}

/// RMS normalization with a fused per-channel gain over the last dimension.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rms_norm_weight(
    a_ptr: i64,
    weight_ptr: i64,
    eps: f64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_RMS_NORM);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_rms_norm_weight");
    if a_ptr == 0 || weight_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let weight = &*(weight_ptr as *const RayzorTensor);
    if a.ndim == 0
        || a.dtype != DTYPE_F32
        || weight.dtype != DTYPE_F32
        || !a.is_contiguous()
        || !weight.is_contiguous()
    {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let last = shape[a.ndim - 1];
    if last == 0 || weight.numel != last {
        return 0;
    }

    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps_f32 = eps as f32;
    let a_data = a.data as *const f32;
    let w_slice = std::slice::from_raw_parts(weight.data as *const f32, last);
    let r_data = r.data as *mut f32;
    for g in 0..groups {
        let base = g * last;
        let a_row = std::slice::from_raw_parts(a_data.add(base), last);
        let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
        rms_norm::rms_norm_row_f32(r_row, a_row, w_slice, eps_f32, f32::sqrt);
    }
    result
}

/// Apply rotary position embedding (RoPE) to a 3-D tensor of shape
/// `[seq_len, num_heads, head_dim]` (or 2-D `[seq_len, head_dim]` with
/// `num_heads = 1`).
///
/// `cos` and `sin` are 2-D tables of shape `[max_seq_len, head_dim / 2]`
/// that were precomputed once for the model's max context length. Only the
/// first `seq_len` rows are consumed; the rest are ignored. `position_offset`
/// adds to the row index — used by the KV-cache decode path when generating
/// token `T` so the new query gets rotated for absolute position `T`.
///
/// The standard Llama rotation acts on adjacent pairs `(x_{2i}, x_{2i+1})`:
/// ```text
///   x_{2i}'   = x_{2i}   * cos[p, i] - x_{2i+1} * sin[p, i]
///   x_{2i+1}' = x_{2i}   * sin[p, i] + x_{2i+1} * cos[p, i]
/// ```
/// where `p` is the absolute position (`row + position_offset`).
///
/// Returns a new tensor with the same shape + dtype.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope(
    x_ptr: i64,
    cos_ptr: i64,
    sin_ptr: i64,
    position_offset: i64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_ROPE);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_rope");
    if x_ptr == 0 || cos_ptr == 0 || sin_ptr == 0 {
        return 0;
    }
    let x = &*(x_ptr as *const RayzorTensor);
    let cos = &*(cos_ptr as *const RayzorTensor);
    let sin = &*(sin_ptr as *const RayzorTensor);

    // Expect at least 2 dims; treat last as head_dim, second-to-last as num_heads,
    // and any leading dim as seq_len (collapsed). Cos/sin must be 2-D
    // [max_seq_len, head_dim/2].
    if x.ndim < 2 || cos.ndim != 2 || sin.ndim != 2 {
        return 0;
    }
    let x_shape = std::slice::from_raw_parts(x.shape, x.ndim);
    let head_dim = x_shape[x.ndim - 1];
    if !head_dim.is_multiple_of(2) {
        return 0;
    }
    let half = head_dim / 2;
    let num_heads = if x.ndim >= 3 { x_shape[x.ndim - 2] } else { 1 };
    let seq_len: usize = x_shape[..x.ndim.saturating_sub(2)]
        .iter()
        .product::<usize>()
        .max(1)
        * (if x.ndim >= 3 { 1 } else { x_shape[0] });
    let cos_shape = std::slice::from_raw_parts(cos.shape, 2);
    let sin_shape = std::slice::from_raw_parts(sin.shape, 2);
    if cos_shape[1] != half || sin_shape[1] != half {
        return 0;
    }
    let cos_max = cos_shape[0];
    let pos_off = position_offset.max(0) as usize;

    let result = alloc_tensor(x_shape, x.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);

    let elements_per_head = head_dim;
    let elements_per_row = num_heads * elements_per_head;

    for s in 0..seq_len {
        let pos = s + pos_off;
        if pos >= cos_max {
            // Position out of range — fall back to identity rotation by copying x.
            for i in 0..elements_per_row {
                let off = s * elements_per_row + i;
                let v = load_f32_at(x.data, off, x.dtype);
                store_f32_at(r.data, off, x.dtype, v);
            }
            continue;
        }
        for h in 0..num_heads {
            for i in 0..half {
                let cos_v = load_f32_at(cos.data, pos * half + i, cos.dtype);
                let sin_v = load_f32_at(sin.data, pos * half + i, sin.dtype);
                let base = s * elements_per_row + h * elements_per_head;
                // GGUF Llama models use the *interleaved* RoPE convention
                // (llama.cpp's GGML_ROPE_TYPE_NORMAL = 0): consecutive
                // dimensions are paired (x[2i], x[2i+1]). The HF
                // `transformers/models/llama/modeling_llama.py::rotate_half`
                // path is half-split (x[i], x[i+half]), but the
                // HF-to-GGUF converter permutes the Q/K weight matrices
                // so the GGUF weights work with the interleaved layout
                // — i.e. the model file already bakes in the convention
                // it expects. Loading those weights and applying half-
                // split RoPE rotates along the wrong pairs of dims, which
                // shows up as a 78% relative error on `Qcur-rope.sum()`
                // vs the llama.cpp reference and a degenerate downstream
                // attention pattern. See ggml-cpu/ops.cpp `ggml_compute_forward_rope`
                // and the `rope type = 0` print_info in `llama-eval-callback -lv 4`.
                let off_lo = base + 2 * i;
                let off_hi = base + 2 * i + 1;
                let xlo = load_f32_at(x.data, off_lo, x.dtype);
                let xhi = load_f32_at(x.data, off_hi, x.dtype);
                store_f32_at(r.data, off_lo, x.dtype, xlo * cos_v - xhi * sin_v);
                store_f32_at(r.data, off_hi, x.dtype, xlo * sin_v + xhi * cos_v);
            }
        }
    }
    result
}

/// Fused flash-style scaled dot-product attention for the **decode** case
/// (seqQ == 1) with grouped-query attention (GQA) built in.
///
/// Replaces the chain
///
/// ```text
///   expandKvHeads(K) + expandKvHeads(V) + bmm(Q, K^T) + scale
///   + causalMask (no-op for decode) + softmax + bmm(attn, V)
/// ```
///
/// with one kernel that streams over the KV cache exactly once. The win
/// comes from memory traffic: at cache_len=568 the chain reads K and V
/// twice (once for the score bmm, once for the context bmm) AFTER
/// materialising 4× expanded copies, ~220 MB/token at 16 layers. The
/// fused kernel reads each KV entry once from the un-expanded cache and
/// never materialises scores — ~40 MB/token.
///
/// Inputs:
///   - `q_ptr` — Q tensor, shape `[1, num_q_heads, head_dim]`, F32,
///     contiguous. The "after-RoPE, before-permute" shape from
///     `GQAttention.forward`. seqQ must equal 1; the kernel is decode-
///     only by design (prefill stays on the bmm chain).
///   - `k_ptr`, `v_ptr` — KV cache **view**, shape
///     `[cache_len, num_kv_heads, head_dim]`, F32. Cache backing is
///     contiguous and the slice-along-axis-0 view preserves the inner
///     two axes' contiguity, so `K[l, h, d] = data[l*num_kv*hd + h*hd + d]`.
///   - `scale` — `1 / sqrt(head_dim)`, applied to each score before
///     softmax. Matches the existing path's `scoresRaw.scale(scale)`.
///
/// Returns a fresh contiguous tensor of shape `[1, num_q_heads, head_dim]`,
/// F32, owning. Returns 0 on any gate violation so the caller can fall
/// back to the bmm chain.
///
/// Numerical match: the kernel uses the standard "max-shifted softmax"
/// for each Q-head, computing scores into a stack array then
/// softmax-weighted V-sum. The reduction order over `cache_len` matches
/// the bmm path (sequential along axis 0), so MATCH-on-canonical
/// should hold modulo a few f32 ULPs at the very tail — not enough to
/// shift argmax on a 128k vocab.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode(
    q_ptr: i64,
    k_ptr: i64,
    v_ptr: i64,
    scale: f64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::FLASH_ATTN_DECODE);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_flash_attn_decode");
    if q_ptr == 0 || k_ptr == 0 || v_ptr == 0 {
        return 0;
    }
    let q = &*(q_ptr as *const RayzorTensor);
    let k = &*(k_ptr as *const RayzorTensor);
    let v = &*(v_ptr as *const RayzorTensor);

    // dtype gate: F32 only for now (the rest of GQAttention is F32).
    if q.dtype != DTYPE_F32 || k.dtype != DTYPE_F32 || v.dtype != DTYPE_F32 {
        return 0;
    }
    // shape gates
    if q.ndim != 3 || k.ndim != 3 || v.ndim != 3 {
        return 0;
    }
    let q_shape = std::slice::from_raw_parts(q.shape, 3);
    let k_shape = std::slice::from_raw_parts(k.shape, 3);
    let v_shape = std::slice::from_raw_parts(v.shape, 3);

    let seq_q = q_shape[0];
    let num_q_heads = q_shape[1];
    let head_dim = q_shape[2];
    let cache_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    // Decode-only by design.
    if seq_q != 1 {
        return 0;
    }
    // Shapes must agree.
    if v_shape[0] != cache_len || v_shape[1] != num_kv_heads || v_shape[2] != head_dim {
        return 0;
    }
    if k_shape[2] != head_dim {
        return 0;
    }
    // GQA group must divide.
    if num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
        return 0;
    }
    let group = num_q_heads / num_kv_heads;

    // Contiguity gate: Q must be contiguous (seq_q=1 makes its layout flat
    // along the head_dim×num_q_heads axes), K/V along the inner two axes —
    // the cache slice view is row-major along (head, dim) so just check that.
    if !q.is_contiguous() {
        return 0;
    }
    let k_strides = std::slice::from_raw_parts(k.strides, 3);
    let v_strides = std::slice::from_raw_parts(v.strides, 3);
    let kv_row_stride = (num_kv_heads * head_dim) as usize;
    if k_strides[1] != head_dim || k_strides[2] != 1 {
        return 0;
    }
    if v_strides[1] != head_dim || v_strides[2] != 1 {
        return 0;
    }
    // The cache-slice view keeps stride[0] = num_kv_heads*head_dim (the
    // original backing's row stride). If something else passes a strided
    // view we bail to avoid scrambled reads.
    if k_strides[0] != kv_row_stride || v_strides[0] != kv_row_stride {
        return 0;
    }

    // Allocate output [1, num_q_heads, head_dim].
    let out_shape = [1usize, num_q_heads, head_dim];
    let result = alloc_tensor(&out_shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);

    let q_data = q.data as *const f32;
    let k_data = k.data as *const f32;
    let v_data = v.data as *const f32;
    let out_data = r.data as *mut f32;

    // Each q_head writes a disjoint head_dim slice of `out_data`
    // (`out_data[q_head * head_dim .. q_head * head_dim + head_dim]`)
    // and reads only Q[q_head] + K[*, kv_head] + V[*, kv_head] — no
    // cross-q_head reduction. So workers can fan out over the q_head
    // axis without synchronisation.
    //
    // Parallelisation gate: short cache_len pays the worker_pool
    // wake/join cost more than the kernel saves. Empirical A/B on
    // M1 Pro (Voronoi long-form, parallel vs single-thread flash):
    //
    //   N=300 (cache ~316):  median +0.4% (sub-noise), thermal
    //                        pairs lose 8% — workers compete with
    //                        throttled matmul for cores
    //   N=600 (cache ~616):  median +12.1%, two paired wins at +14%
    //
    // Crossover is around cache_len 400-500. Gate at cache_len ≥ 256
    // so the per-q_head work (2 × cache_len × head_dim FMAs ≈ 32k
    // for head_dim=64) is large enough to amortise the fork-join
    // and stays out of the thermally-fragile short-cache regime.
    // Below the gate, single-thread is both faster on cool runs
    // AND more thermally stable.
    let auto_threads: usize = crate::worker_pool::auto_kernel_threads();
    let mut t = auto_threads.min(num_q_heads);
    if cache_len < 256 || num_q_heads < 4 {
        t = 1;
    }
    let scale_f32 = scale as f32;
    if t <= 1 {
        // Sequential fallback. Single scores scratch shared across q_heads.
        let mut scores: Vec<f32> = vec![0.0; cache_len];
        for q_head in 0..num_q_heads {
            flash_attn_decode_one_qhead(
                q_head,
                group,
                head_dim,
                cache_len,
                kv_row_stride,
                scale_f32,
                q_data,
                k_data,
                v_data,
                out_data,
                &mut scores,
            );
        }
        return result;
    }

    // Parallel fan-out. SAFETY: the raw `*const f32` / `*mut f32`
    // pointers don't implement `Send`, so capture as `usize` and
    // reconstitute inside the worker closure. Disjoint writes on
    // `out_data` are guaranteed by the q_head split (each q_head
    // owns a unique `[q_head * head_dim, (q_head+1) * head_dim)`
    // band). Worker_pool::parallel_rows blocks until all jobs
    // finish, so the borrowed pointers stay valid throughout.
    let q_data_us = q_data as usize;
    let k_data_us = k_data as usize;
    let v_data_us = v_data as usize;
    let out_data_us = out_data as usize;
    crate::worker_pool::global().parallel_rows(num_q_heads, t, move |lo, hi| unsafe {
        let q_ptr = q_data_us as *const f32;
        let k_ptr = k_data_us as *const f32;
        let v_ptr = v_data_us as *const f32;
        let out_ptr = out_data_us as *mut f32;
        // Per-worker scratch — reused across the worker's q_head
        // range. One alloc per worker per call (~4 allocs total at
        // t=6 worker spawn instead of 1; the cost amortises against
        // the saved single-thread serialization).
        let mut scores: Vec<f32> = vec![0.0; cache_len];
        for q_head in lo..hi {
            flash_attn_decode_one_qhead(
                q_head,
                group,
                head_dim,
                cache_len,
                kv_row_stride,
                scale_f32,
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                &mut scores,
            );
        }
    });

    result
}

/// Inner per-q_head body shared between the sequential and parallel
/// dispatch paths of `rayzor_tensor_flash_attn_decode`.
///
/// The math lives in `rayzor_runtime_core::tensor::flash_attn`; this thin
/// wrapper picks `f32::exp` (Accelerate-routed on macOS, fast libm
/// elsewhere) as the softmax exponential. The WASM crate uses the same
/// runtime-core kernel with `libm::expf` — see
/// `docs/design/runtime_core_extraction.md` and
/// `docs/design/wasm_runtime_parity.md`.
#[inline]
#[allow(clippy::too_many_arguments)] // hot decode kernel; bundling into a struct would add a load on the inner loop
unsafe fn flash_attn_decode_one_qhead(
    q_head: usize,
    group: usize,
    head_dim: usize,
    cache_len: usize,
    kv_row_stride: usize,
    scale_f32: f32,
    q_data: *const f32,
    k_data: *const f32,
    v_data: *const f32,
    out_data: *mut f32,
    scores: &mut [f32],
) {
    rayzor_runtime_core::tensor::flash_attn::flash_attn_decode_one_qhead(
        q_head,
        group,
        head_dim,
        cache_len,
        kv_row_stride,
        scale_f32,
        q_data,
        k_data,
        v_data,
        out_data,
        scores,
        |x| x.exp(),
    );
}

/// Generate the RoPE cos/sin tables for a given head dimension and maximum
/// sequence length. Returns the cos table; the sin table is generated by
/// the sibling `rayzor_tensor_rope_sin_table`. Both share the same shape
/// `[max_seq_len, head_dim / 2]` and dtype F32.
///
/// The frequency schedule is the Llama / GPT-NeoX standard:
/// ```text
///   theta_i = 1.0 / (base ^ (2i / head_dim))   for i in 0..head_dim/2
///   cos[p, i] = cos(p * theta_i)
///   sin[p, i] = sin(p * theta_i)
/// ```
/// `base` defaults to 10000.0 in most Llama checkpoints (passed as f64 from
/// Haxe). Use the same base used by the model that produced your weights.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, /* sin */ false)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, /* sin */ true)
}

/// F16-stored variants of the RoPE LUTs. Same math, half the memory.
/// Useful when the host wants to keep both tables resident in a tight
/// VRAM budget (Llama 3.2 1B with `max_seq_len=2048` and `head_dim=64`
/// is 256 KB at F32, 128 KB at F16 — the savings amortise into the
/// rest of the activation budget on WebGPU buffer caps).
///
/// Precision note: the rotation step inside the kernel converts each
/// LUT element back to f32 before the multiply, so the precision loss
/// is bounded by the F16 quantisation of `cos / sin ∈ [-1, 1]` —
/// roughly 5e-4 absolute. Indistinguishable in practice for inference.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table_f16(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, /* sin */ false)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table_f16(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, /* sin */ true)
}

unsafe fn rope_table(head_dim: i64, max_seq_len: i64, base: f64, dtype: u8, want_sin: bool) -> i64 {
    if head_dim <= 0 || max_seq_len <= 0 || head_dim % 2 != 0 {
        return 0;
    }
    let head_dim = head_dim as usize;
    let max_seq_len = max_seq_len as usize;
    let half = head_dim / 2;
    let shape = [max_seq_len, half];
    let result = alloc_tensor(&shape, dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let head_dim_f = head_dim as f64;
    for p in 0..max_seq_len {
        for i in 0..half {
            let theta = 1.0_f64 / base.powf((2 * i) as f64 / head_dim_f);
            let angle = (p as f64) * theta;
            let v = if want_sin { angle.sin() } else { angle.cos() };
            store_f32_at(r.data, p * half + i, dtype, v as f32);
        }
    }
    result
}

/// Batched 3-D matmul: `a [batch, M, K]` × `b [batch, K, N]` → `[batch, M, N]`.
/// Per-batch independent matmul; reuses the existing F32 axpy fast path
/// when the inputs are contiguous F32, scalar fallback for other dtypes.
///
/// This is the core kernel that lets the Haxe layer build attention as
/// `(Q @ Kᵀ) → softmax → (· V)` without materialising a per-head loop in
/// user code.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_bmm(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.ndim != 3 || b.ndim != 3 || a.dtype != b.dtype {
        return 0;
    }
    let a_shape = std::slice::from_raw_parts(a.shape, 3);
    let b_shape = std::slice::from_raw_parts(b.shape, 3);
    let a_strides = std::slice::from_raw_parts(a.strides, 3);
    let b_strides = std::slice::from_raw_parts(b.strides, 3);
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];
    if b_shape[0] != batch || b_shape[1] != k {
        return 0;
    }
    let out_shape = [batch, m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let dtype = a.dtype;

    // Result is contiguous (freshly allocated). Walk per batch using
    // each input's *actual* strides — bmm callers in the transformer hot
    // path (GQAttention's qByHead.bmm(kT) and attn.bmm(vAllExpanded)) feed
    // non-contiguous views from `.permute` / `.transposeLast2`. Previously
    // bmm assumed `[batch, M, K]` row-major contiguous and read the wrong
    // memory; attention scores were garbage, leading to incoherent
    // generation regardless of how correct the rest of the pipeline was.
    let a_b_stride = a_strides[0];
    let a_m_stride = a_strides[1];
    let a_k_stride = a_strides[2];
    let b_b_stride = b_strides[0];
    let b_k_stride = b_strides[1];
    let b_n_stride = b_strides[2];

    let a_contig_inner = a_k_stride == 1;
    let b_contig_inner = b_n_stride == 1;

    for batch_i in 0..batch {
        let a_batch_off = batch_i * a_b_stride;
        let b_batch_off = batch_i * b_b_stride;
        let c_batch_off = batch_i * m * n;

        if dtype == DTYPE_F32 && a_contig_inner && b_contig_inner {
            // Inner dim contiguous on both → SIMD axpy fast path.
            let a_f = a.data as *const f32;
            let b_f = b.data as *const f32;
            let c_f = r.data as *mut f32;
            for i in 0..m {
                let c_row_off = c_batch_off + i * n;
                let a_row_off = a_batch_off + i * a_m_stride;
                for p in 0..k {
                    let a_ik = *a_f.add(a_row_off + p);
                    let b_row_off = b_batch_off + p * b_k_stride;
                    let c_slice = std::slice::from_raw_parts_mut(c_f.add(c_row_off), n);
                    let b_slice = std::slice::from_raw_parts(b_f.add(b_row_off), n);
                    crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
                }
            }
            continue;
        }

        // General strided / non-F32 path.
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av =
                        load_f32_at(a.data, a_batch_off + i * a_m_stride + p * a_k_stride, dtype);
                    let bv =
                        load_f32_at(b.data, b_batch_off + p * b_k_stride + j * b_n_stride, dtype);
                    acc += av * bv;
                }
                store_f32_at(r.data, c_batch_off + i * n + j, dtype, acc);
            }
        }
    }
    result
}

/// Threaded variant of `rayzor_tensor_bmm`. Same `[batch, M, K] @ [batch, K, N]
/// -> [batch, M, N]` contract; same stride-aware kernel for the
/// `.permute` / `.transposeLast2` views the transformer attention path
/// feeds in. Parallelises across the flattened `(batch, M)` row space so
/// every worker writes a disjoint contiguous slab of the output.
///
/// `threads`: `0` selects the auto count (6, mirroring
/// `matmulXTQThreaded` — empirically the sweet spot on M1 Pro), `1`
/// short-circuits to the sequential fast path, otherwise clamped to
/// `min(threads, 64)`. F32 only for now; other dtypes return `0`
/// rather than silently falling through.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_bmm_threaded(a_ptr: i64, b_ptr: i64, threads: i64) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_bmm_threaded");
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.ndim != 3 || b.ndim != 3 || a.dtype != b.dtype {
        return 0;
    }
    if a.dtype != DTYPE_F32 {
        return 0;
    }
    let a_shape = std::slice::from_raw_parts(a.shape, 3);
    let b_shape = std::slice::from_raw_parts(b.shape, 3);
    let a_strides = std::slice::from_raw_parts(a.strides, 3);
    let b_strides = std::slice::from_raw_parts(b.strides, 3);
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];
    if b_shape[0] != batch || b_shape[1] != k {
        return 0;
    }
    let out_shape = [batch, m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let dtype = a.dtype;

    let a_b_stride = a_strides[0];
    let a_m_stride = a_strides[1];
    let a_k_stride = a_strides[2];
    let b_b_stride = b_strides[0];
    let b_k_stride = b_strides[1];
    let b_n_stride = b_strides[2];

    let auto_threads: usize = crate::worker_pool::auto_kernel_threads();
    let total_rows = batch * m;
    let mut t = if threads > 0 {
        (threads as usize).min(64)
    } else {
        auto_threads
    };
    if t > total_rows {
        t = total_rows.max(1);
    }

    // Sequential fast path: skip fork/join when work is too small to amortize.
    // ~64 rows is the empirical break-even on M1 Pro with parallel_rows spawn
    // cost (each worker needs >=~10 rows worth of FMA to dominate the join).
    const MIN_PARALLEL_ROWS: usize = 64;
    if t <= 1 || total_rows < MIN_PARALLEL_ROWS {
        let a_contig_inner = a_k_stride == 1;
        let b_contig_inner = b_n_stride == 1;
        for batch_i in 0..batch {
            let a_batch_off = batch_i * a_b_stride;
            let b_batch_off = batch_i * b_b_stride;
            let c_batch_off = batch_i * m * n;
            if a_contig_inner && b_contig_inner {
                let a_f = a.data as *const f32;
                let b_f = b.data as *const f32;
                let c_f = r.data as *mut f32;
                for i in 0..m {
                    let c_row_off = c_batch_off + i * n;
                    let a_row_off = a_batch_off + i * a_m_stride;
                    for p in 0..k {
                        let a_ik = *a_f.add(a_row_off + p);
                        let b_row_off = b_batch_off + p * b_k_stride;
                        let c_slice = std::slice::from_raw_parts_mut(c_f.add(c_row_off), n);
                        let b_slice = std::slice::from_raw_parts(b_f.add(b_row_off), n);
                        crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
                    }
                }
            } else {
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            let av = load_f32_at(
                                a.data,
                                a_batch_off + i * a_m_stride + p * a_k_stride,
                                dtype,
                            );
                            let bv = load_f32_at(
                                b.data,
                                b_batch_off + p * b_k_stride + j * b_n_stride,
                                dtype,
                            );
                            acc += av * bv;
                        }
                        store_f32_at(r.data, c_batch_off + i * n + j, dtype, acc);
                    }
                }
            }
        }
        return result;
    }

    let a_data = a.data as usize;
    let b_data = b.data as usize;
    let r_data = r.data as usize;
    let m_dim = m;
    let n_dim = n;
    let k_dim = k;
    let a_contig_inner = a_k_stride == 1;
    let b_contig_inner = b_n_stride == 1;

    crate::worker_pool::global().parallel_rows(total_rows, t, move |lo, hi| {
        // SAFETY: each worker writes Y[batch_i, m_i, 0..N] for the (batch_i, m_i)
        // pairs it owns; ranges are disjoint across workers so there is no
        // aliasing on the output. Inputs A and B are read-only.
        unsafe {
            for flat in lo..hi {
                let batch_i = flat / m_dim;
                let m_i = flat % m_dim;
                let a_batch_off = batch_i * a_b_stride;
                let b_batch_off = batch_i * b_b_stride;
                let c_batch_off = batch_i * m_dim * n_dim;
                let c_row_off = c_batch_off + m_i * n_dim;
                let a_row_off = a_batch_off + m_i * a_m_stride;
                if a_contig_inner && b_contig_inner {
                    let a_f = a_data as *const f32;
                    let b_f = b_data as *const f32;
                    let c_f = r_data as *mut f32;
                    for p in 0..k_dim {
                        let a_ik = *a_f.add(a_row_off + p);
                        let b_row_off = b_batch_off + p * b_k_stride;
                        let c_slice = std::slice::from_raw_parts_mut(c_f.add(c_row_off), n_dim);
                        let b_slice = std::slice::from_raw_parts(b_f.add(b_row_off), n_dim);
                        crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
                    }
                } else {
                    let a_ptr = a_data as *const u8;
                    let b_ptr = b_data as *const u8;
                    let r_ptr = r_data as *mut u8;
                    for j in 0..n_dim {
                        let mut acc = 0.0f32;
                        for p in 0..k_dim {
                            let av = load_f32_at(
                                a_ptr,
                                a_batch_off + m_i * a_m_stride + p * a_k_stride,
                                dtype,
                            );
                            let bv = load_f32_at(
                                b_ptr,
                                b_batch_off + p * b_k_stride + j * b_n_stride,
                                dtype,
                            );
                            acc += av * bv;
                        }
                        store_f32_at(r_ptr, c_row_off + j, dtype, acc);
                    }
                }
            }
        }
    });

    result
}

/// In-place causal mask. Treats the last two dimensions of `t` as
/// `[..., rows, cols]` and fills positions `(i, j)` with `-inf` whenever
/// `j > i + position_offset`. Standard pattern: a softmax row reads the
/// masked positions as zero probability.
///
/// `position_offset` shifts the diagonal — 0 for prefill (every query row
/// sees keys up to its own index), positive for decode (the single new
/// query at logical position T attends to keys 0..=T).
///
/// Returns the same tensor pointer (mutates in place) for convenience.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_causal_mask_(t_ptr: i64, position_offset: i64) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_causal_mask_");
    if t_ptr == 0 {
        return 0;
    }
    let t = &*(t_ptr as *const RayzorTensor);
    if t.ndim < 2 {
        return 0;
    }
    let shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let cols = shape[t.ndim - 1];
    let rows = shape[t.ndim - 2];
    let outer: usize = shape[..t.ndim.saturating_sub(2)]
        .iter()
        .product::<usize>()
        .max(1);
    let pos = position_offset.max(0) as usize;
    let neg_inf = f32::NEG_INFINITY;
    for o in 0..outer {
        let base = o * rows * cols;
        for i in 0..rows {
            // Mask everything strictly after the diagonal (+ position_offset).
            let first_masked = (i + pos + 1).min(cols);
            for j in first_masked..cols {
                store_f32_at(t.data, base + i * cols + j, t.dtype, neg_inf);
            }
        }
    }
    t_ptr
}

/// Scale every element by a scalar f32. Allocates a fresh tensor; no
/// in-place variant since composing with other ops works just as well
/// after the new allocation.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_scale(t_ptr: i64, factor: f64) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_scale");
    if t_ptr == 0 {
        return 0;
    }
    let t = &*(t_ptr as *const RayzorTensor);
    let shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let result = alloc_tensor(shape, t.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let f = factor as f32;
    if t.dtype == DTYPE_F32 {
        let src = t.data as *const f32;
        let dst = r.data as *mut f32;
        let n = t.numel;
        let src_slice = std::slice::from_raw_parts(src, n);
        let dst_slice = std::slice::from_raw_parts_mut(dst, n);
        crate::tensor_simd::mul_const_slice(dst_slice, src_slice, f);
        return result;
    }
    for i in 0..t.numel {
        let v = load_f32_at(t.data, i, t.dtype);
        store_f32_at(r.data, i, t.dtype, v * f);
    }
    result
}

/// Transpose the last two dimensions (zero-copy view). Equivalent to
/// `tensor.permute([..., ndim-1, ndim-2])` for ndim ≥ 2. The common use is
/// turning `K [seq_k, num_kv_heads, head_dim]` into something we can
/// matmul against, but for true 3-D batched matmul on K we'd more often
/// pre-transpose at the per-head level — depends on layout choices.
/// Provided for completeness so nue doesn't need to reach for permute()
/// every time.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_transpose_last2(t_ptr: i64) -> i64 {
    if t_ptr == 0 {
        return 0;
    }
    let t = &*(t_ptr as *const RayzorTensor);
    if t.ndim < 2 {
        return t_ptr;
    }
    let n = t.ndim;
    let old_shape = std::slice::from_raw_parts(t.shape, n);
    let old_strides = std::slice::from_raw_parts(t.strides, n);

    let new_shape_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }
    for i in 0..n {
        *new_shape_ptr.add(i) = old_shape[i];
        *new_strides_ptr.add(i) = old_strides[i];
    }
    // Swap the last two.
    *new_shape_ptr.add(n - 1) = old_shape[n - 2];
    *new_shape_ptr.add(n - 2) = old_shape[n - 1];
    *new_strides_ptr.add(n - 1) = old_strides[n - 2];
    *new_strides_ptr.add(n - 2) = old_strides[n - 1];

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        free(new_shape_ptr as *mut u8);
        free(new_strides_ptr as *mut u8);
        return 0;
    }
    *new_t = RayzorTensor {
        data: t.data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: n,
        numel: t.numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: t_ptr as *mut RayzorTensor,
    };
    t.refcount
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    new_t as i64
}

/// Row-gather: fetch the rows of `table` named by `indices` and stack
/// them into a new tensor.
///
/// `table` is `[N, ...rest]`; `indices` is an i64 array of length `K`.
/// The result has shape `[K, ...rest]` and shares the source dtype.
/// Used by `nue.Embedding` to turn `[seq_len]` token IDs into a
/// `[seq_len, hidden_size]` activation tensor.
///
/// Out-of-range indices return 0 — caller is responsible for validating
/// the vocabulary range. The indices array is read as i64, matching the
/// Haxe Array<Int> layout (which boxes ints to i64 in this runtime).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows(
    table_ptr: i64,
    indices_ptr: i64,
    indices_len: i64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_GATHER_ROWS);
    if table_ptr == 0 || indices_ptr == 0 || indices_len <= 0 {
        return 0;
    }
    let table = &*(table_ptr as *const RayzorTensor);
    if table.ndim == 0 {
        return 0;
    }
    let table_shape = std::slice::from_raw_parts(table.shape, table.ndim);
    let n_rows = table_shape[0];
    let row_numel: usize = table_shape[1..].iter().product::<usize>().max(1);
    let elem = dtype_size(table.dtype);
    let row_bytes = row_numel * elem;
    let k = indices_len as usize;

    // Out shape: [K, ...table_shape[1..]]
    let mut out_shape = Vec::with_capacity(table.ndim);
    out_shape.push(k);
    for &dim in &table_shape[1..] {
        out_shape.push(dim);
    }
    let result = alloc_tensor(&out_shape, table.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let indices = indices_ptr as *const i64;

    for i in 0..k {
        let idx_raw = *indices.add(i);
        if idx_raw < 0 || (idx_raw as usize) >= n_rows {
            // Out of range — leave the corresponding output row zeroed.
            continue;
        }
        let src = table.data.add((idx_raw as usize) * row_bytes);
        let dst = r.data.add(i * row_bytes);
        std::ptr::copy_nonoverlapping(src, dst, row_bytes);
    }
    result
}

// ============================================================================
// Reductions
// ============================================================================

/// tensor.sum() -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sum(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.dtype == DTYPE_F32 {
        let data = std::slice::from_raw_parts(t.data as *const f32, t.numel);
        return crate::tensor_simd::sum_slice(data) as f64;
    }
    let mut acc = 0.0f64;
    for i in 0..t.numel {
        acc += load_f32_at(t.data, i, t.dtype) as f64;
    }
    acc
}

/// tensor.max() -> f64 (returns -inf for empty tensors)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_max(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return f64::NEG_INFINITY;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.numel == 0 {
        return f64::NEG_INFINITY;
    }
    if t.dtype == DTYPE_F32 {
        let data = std::slice::from_raw_parts(t.data as *const f32, t.numel);
        return crate::tensor_simd::max_slice(data) as f64;
    }
    let mut m = f32::NEG_INFINITY;
    for i in 0..t.numel {
        let v = load_f32_at(t.data, i, t.dtype);
        if v > m {
            m = v;
        }
    }
    m as f64
}

/// tensor.min() -> f64 (returns +inf for empty tensors)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_min(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return f64::INFINITY;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.numel == 0 {
        return f64::INFINITY;
    }
    if t.dtype == DTYPE_F32 {
        let data = std::slice::from_raw_parts(t.data as *const f32, t.numel);
        return crate::tensor_simd::min_slice(data) as f64;
    }
    let mut m = f32::INFINITY;
    for i in 0..t.numel {
        let v = load_f32_at(t.data, i, t.dtype);
        if v < m {
            m = v;
        }
    }
    m as f64
}

/// tensor.mean() -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mean(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.numel == 0 {
        return 0.0;
    }
    rayzor_tensor_sum(tensor_ptr) / (t.numel as f64)
}

/// tensor.dot(other) -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_dot(a_ptr: i64, b_ptr: i64) -> f64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0.0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.numel != b.numel || a.dtype != b.dtype {
        return 0.0;
    }

    if a.dtype == DTYPE_F32 {
        let a_s = std::slice::from_raw_parts(a.data as *const f32, a.numel);
        let b_s = std::slice::from_raw_parts(b.data as *const f32, b.numel);
        return crate::tensor_simd::dot_slice(a_s, b_s) as f64;
    }
    let mut acc = 0.0f64;
    for i in 0..a.numel {
        let av = load_f32_at(a.data, i, a.dtype) as f64;
        let bv = load_f32_at(b.data, i, a.dtype) as f64;
        acc += av * bv;
    }
    acc
}

// ============================================================================
// Matrix multiplication
// ============================================================================

/// tensor.matmul(other) -> i64
/// Naive O(n³) matmul for [M,K] × [K,N] -> [M,N]
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul(a_ptr: i64, b_ptr: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL);
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    if a.ndim != 2 || b.ndim != 2 || a.dtype != b.dtype {
        return 0;
    }

    let a_shape = std::slice::from_raw_parts(a.shape, 2);
    let b_shape = std::slice::from_raw_parts(b.shape, 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return 0;
    } // dimension mismatch

    let out_shape = [m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_strides = std::slice::from_raw_parts(a.strides, 2);
    let b_strides = std::slice::from_raw_parts(b.strides, 2);

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let b_data = b.data as *const f32;
        let r_data = r.data as *mut f32;

        // Fast path: both A and B row-major (innermost stride == 1). Loop
        // order is (i, k, j) so the inner `j` loop accumulates into a
        // contiguous row of R with broadcast a_ik — the textbook
        // SIMD-friendly matmul.
        if a_strides[1] == 1 && b_strides[1] == 1 {
            let r_row_size = n;
            for i in 0..m {
                let a_row = a_data.add(i * a_strides[0]);
                let r_row = r_data.add(i * r_row_size);
                for p in 0..k {
                    let a_ik = *a_row.add(p);
                    let b_row = b_data.add(p * b_strides[0]);
                    let r_slice = std::slice::from_raw_parts_mut(r_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_slice(r_slice, a_ik, b_slice);
                }
            }
            return result;
        }

        // Strided fallback (e.g. transposed views).
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = *a_data.add(i * a_strides[0] + p * a_strides[1]);
                    let b_val = *b_data.add(p * b_strides[0] + j * b_strides[1]);
                    sum += a_val * b_val;
                }
                *r_data.add(i * n + j) = sum;
            }
        }
        return result;
    }

    // F16 / BF16 row-major fast path: axpy specialisation that stages
    // through f32 via NEON vcvt_f32_f16 / F16C _mm_cvtph_ps in batches of 64.
    // This is the matmul hot path for LLM inference (every row update is an
    // f16 axpy against a half-precision weight row).
    if a_strides[1] == 1 && b_strides[1] == 1 {
        if a.dtype == DTYPE_F16 {
            let a_data = a.data as *const u16;
            let b_data = b.data as *const u16;
            let r_data = r.data as *mut u16;
            for i in 0..m {
                let a_row = a_data.add(i * a_strides[0]);
                let r_row = r_data.add(i * n);
                for p in 0..k {
                    let a_ik = half::f16::from_bits(*a_row.add(p)).to_f32();
                    let b_row = b_data.add(p * b_strides[0]);
                    let r_slice = std::slice::from_raw_parts_mut(r_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_f16_slice(r_slice, a_ik, b_slice);
                }
            }
            return result;
        }
        if a.dtype == DTYPE_BF16 {
            let a_data = a.data as *const u16;
            let b_data = b.data as *const u16;
            let r_data = r.data as *mut u16;
            for i in 0..m {
                let a_row = a_data.add(i * a_strides[0]);
                let r_row = r_data.add(i * n);
                for p in 0..k {
                    let a_ik = half::bf16::from_bits(*a_row.add(p)).to_f32();
                    let b_row = b_data.add(p * b_strides[0]);
                    let r_slice = std::slice::from_raw_parts_mut(r_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_bf16_slice(r_slice, a_ik, b_slice);
                }
            }
            return result;
        }
    }

    // Generic dtype path: convert each element to f32 in-register, accumulate
    // in f32, store back as the source dtype. Covers strided F16/BF16 views
    // and the integer/FP8 dtypes.
    let dtype = a.dtype;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_off = i * a_strides[0] + p * a_strides[1];
                let b_off = p * b_strides[0] + j * b_strides[1];
                let a_val = load_f32_at(a.data, a_off, dtype);
                let b_val = load_f32_at(b.data, b_off, dtype);
                sum += a_val * b_val;
            }
            store_f32_at(r.data, i * n + j, dtype, sum);
        }
    }

    result
}

/// Matmul with transposed RHS: `y[i, j] = sum_k a[i, k] * b[j, k]`.
///
/// A is `[M, K]`, B is `[N, K]` (logically transposed — its second dim is
/// the K of matmul). Result is `[M, N]`. This is the natural shape for
/// PyTorch-style `Linear`: `y = x @ w.T` with `w[out, in]` and
/// `x[batch, in]`.
///
/// Iteration loops as (i, j, k) — the innermost k is a dot product of two
/// contiguous rows when both A and B are row-major (their row strides are
/// each non-unit but their column strides are 1). That's SIMD-friendly:
/// the inner loop becomes a fused-multiply-add reduction. Compared to
/// `rayzor_tensor_matmul` (which uses axpy along columns of B), this
/// avoids the strided B access entirely.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_t(a_ptr: i64, b_ptr: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL_T);
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    if a.ndim != 2 || b.ndim != 2 || a.dtype != b.dtype {
        return 0;
    }

    let a_shape = std::slice::from_raw_parts(a.shape, 2);
    let b_shape = std::slice::from_raw_parts(b.shape, 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    if k != b_shape[1] {
        return 0;
    }

    let out_shape = [m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_strides = std::slice::from_raw_parts(a.strides, 2);
    let b_strides = std::slice::from_raw_parts(b.strides, 2);

    if a.dtype == DTYPE_F32 && a_strides[1] == 1 && b_strides[1] == 1 {
        let a_data = a.data as *const f32;
        let b_data = b.data as *const f32;
        let r_data = r.data as *mut f32;
        for i in 0..m {
            let a_row = std::slice::from_raw_parts(a_data.add(i * a_strides[0]), k);
            for j in 0..n {
                let b_row = std::slice::from_raw_parts(b_data.add(j * b_strides[0]), k);
                let sum = crate::tensor_simd::dot_slice_f32(a_row, b_row);
                *r_data.add(i * n + j) = sum;
            }
        }
        return result;
    }

    // Generic fallback for non-F32 / strided inputs.
    let dtype = a.dtype;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_off = i * a_strides[0] + p * a_strides[1];
                let b_off = j * b_strides[0] + p * b_strides[1];
                let a_val = load_f32_at(a.data, a_off, dtype);
                let b_val = load_f32_at(b.data, b_off, dtype);
                sum += a_val * b_val;
            }
            store_f32_at(r.data, i * n + j, dtype, sum);
        }
    }

    result
}

/// Threaded variant of `rayzor_tensor_matmul_t`. Same `[M, K] @ [N, K] -> [M, N]`
/// contract; same scalar dot-product kernel per `(i, j)` cell so the
/// floating-point reduction order is byte-identical to the sequential symbol.
/// Parallelism only fans out across the `M` (output-row) axis — each output
/// row's `k` reduction stays on a single worker, so workers never share a
/// partial sum.
///
/// `threads`: `0` selects the auto count (6, mirroring `bmm_threaded` /
/// `matmulXTQThreaded` — the empirical sweet spot on M1 Pro), `1` short-circuits
/// to the sequential fast path, otherwise clamped to `min(threads, 64)`. When
/// `M` is below `MIN_PARALLEL_ROWS` we skip fork/join overhead and inline the
/// sequential body.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_t_threaded(
    a_ptr: i64,
    b_ptr: i64,
    threads: i64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL_T_THREADED);
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    if a.ndim != 2 || b.ndim != 2 || a.dtype != b.dtype {
        return 0;
    }

    let a_shape = std::slice::from_raw_parts(a.shape, 2);
    let b_shape = std::slice::from_raw_parts(b.shape, 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    if k != b_shape[1] {
        return 0;
    }

    let out_shape = [m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_strides = std::slice::from_raw_parts(a.strides, 2);
    let b_strides = std::slice::from_raw_parts(b.strides, 2);
    let dtype = a.dtype;
    let a_row_stride = a_strides[0];
    let a_col_stride = a_strides[1];
    let b_row_stride = b_strides[0];
    let b_col_stride = b_strides[1];
    let f32_contig = dtype == DTYPE_F32 && a_col_stride == 1 && b_col_stride == 1;

    let auto_threads: usize = crate::worker_pool::auto_kernel_threads();
    let mut t = if threads > 0 {
        (threads as usize).min(64)
    } else {
        auto_threads
    };
    if t > m {
        t = m.max(1);
    }

    // Sequential fast path: skip fork/join when work is too small to amortize.
    // ~64 rows is the empirical break-even on M1 Pro with parallel_rows spawn
    // cost (mirrors `rayzor_tensor_bmm_threaded`).
    const MIN_PARALLEL_ROWS: usize = 64;
    if t <= 1 || m < MIN_PARALLEL_ROWS {
        if f32_contig {
            let a_data = a.data as *const f32;
            let b_data = b.data as *const f32;
            let r_data = r.data as *mut f32;
            for i in 0..m {
                let a_row = std::slice::from_raw_parts(a_data.add(i * a_row_stride), k);
                for j in 0..n {
                    let b_row = std::slice::from_raw_parts(b_data.add(j * b_row_stride), k);
                    let sum = crate::tensor_simd::dot_slice_f32(a_row, b_row);
                    *r_data.add(i * n + j) = sum;
                }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        let a_off = i * a_row_stride + p * a_col_stride;
                        let b_off = j * b_row_stride + p * b_col_stride;
                        let a_val = load_f32_at(a.data, a_off, dtype);
                        let b_val = load_f32_at(b.data, b_off, dtype);
                        sum += a_val * b_val;
                    }
                    store_f32_at(r.data, i * n + j, dtype, sum);
                }
            }
        }
        return result;
    }

    let a_data = a.data as usize;
    let b_data = b.data as usize;
    let r_data = r.data as usize;
    let m_dim = m;
    let n_dim = n;
    let k_dim = k;

    crate::worker_pool::global().parallel_rows(m_dim, t, move |lo, hi| {
        // SAFETY: each worker writes Y[i, 0..N] for the `i` rows in its band;
        // bands are disjoint so there is no aliasing on the output. Inputs A
        // and B are read-only. The scalar reduction per (i, j) stays inside a
        // single worker, so the f32 accumulation order matches the sequential
        // `rayzor_tensor_matmul_t` body byte-for-byte.
        unsafe {
            if f32_contig {
                let a_f = a_data as *const f32;
                let b_f = b_data as *const f32;
                let c_f = r_data as *mut f32;
                for i in lo..hi {
                    let a_row = std::slice::from_raw_parts(a_f.add(i * a_row_stride), k_dim);
                    for j in 0..n_dim {
                        let b_row = std::slice::from_raw_parts(b_f.add(j * b_row_stride), k_dim);
                        let sum = crate::tensor_simd::dot_slice_f32(a_row, b_row);
                        *c_f.add(i * n_dim + j) = sum;
                    }
                }
            } else {
                let a_ptr = a_data as *const u8;
                let b_ptr = b_data as *const u8;
                let r_ptr = r_data as *mut u8;
                for i in lo..hi {
                    for j in 0..n_dim {
                        let mut sum = 0.0f32;
                        for p in 0..k_dim {
                            let a_off = i * a_row_stride + p * a_col_stride;
                            let b_off = j * b_row_stride + p * b_col_stride;
                            let a_val = load_f32_at(a_ptr, a_off, dtype);
                            let b_val = load_f32_at(b_ptr, b_off, dtype);
                            sum += a_val * b_val;
                        }
                        store_f32_at(r_ptr, i * n_dim + j, dtype, sum);
                    }
                }
            }
        }
    });

    result
}

// ============================================================================
// Interop
// ============================================================================

/// tensor.data() -> i64 (raw pointer to data buffer)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_data(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.data as i64
}

/// Atomic-refcount clone: bump `src`'s refcount and return the same pointer.
///
/// Phase 1 ARC semantic. Every `@:derive([Clone])` Tensor call site on the
/// Haxe side audited as an alias-after-move workaround (Linear / GQAttention /
/// TransformerBlock all clone purely to satisfy the linearised use-of-moved
/// analyzer — the matmul kernels never mutate their inputs), so flipping
/// `.clone()` from deep-copy to refcount-bump is safe across the existing
/// callsites and turns each clone from `4 mallocs + O(numel) memcpy` into a
/// single Relaxed fetch_add.
///
/// Disjoint-storage callers should use `rayzor_tensor_deep_clone` instead
/// (kept as the moral equivalent of the old `rayzor_tensor_clone` body,
/// including compact-to-contiguous on views).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_arc_clone(src: i64) -> i64 {
    if src == 0 {
        return 0;
    }
    let s = &*(src as *const RayzorTensor);
    // Relaxed: matches Boost intrusive_ptr — the increment doesn't need to
    // observe anything published before it; the AcqRel pairing only matters
    // on the decrement-to-zero gate in `rayzor_tensor_free`.
    s.refcount
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    src
}

/// `Tensor.clone(src)` Haxe entry point. Routes to the Arc-increment path
/// (see `rayzor_tensor_arc_clone`). The original `rayzor_tensor_clone`
/// extern name is preserved for ABI compatibility with the Tier B
/// `@:derive([Clone])` lowering in hir_to_mir.rs.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_clone(src: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_CLONE);
    rayzor_tensor_arc_clone(src)
}

/// Disjoint-storage deep clone. Materialises a fresh, fully-owning,
/// **contiguous** tensor sharing no storage with `src`.
///
/// This is the historical body of `rayzor_tensor_clone`, kept as an escape
/// hatch for the small number of future callers that genuinely need
/// disjoint backing storage (e.g. saving a pre-`addInto` snapshot, or
/// writing into the buffer across threads without lock contention).
///
/// - `src == 0` → returns 0 (null pass-through).
/// - Both owning tensors AND views are deep-copied. There is no pass-through
///   for views: returning `src` for `!owns_data` aliased the parent's
///   shape/strides/wrapper.
/// - The result owns `numel * dtype_size(dtype)` bytes of data, freshly
///   malloc'd, with canonical row-major strides.
/// - If `src` is already contiguous, the body is a single `memcpy`.
/// - If `src` is non-contiguous (permute/slice/transpose view), we walk by
///   strides and gather byte-block-by-byte-block into a fresh row-major
///   contiguous buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_deep_clone(src: i64) -> i64 {
    if src == 0 {
        return 0;
    }
    let s = &*(src as *const RayzorTensor);

    let elem_size = dtype_size(s.dtype);
    let ndim = s.ndim;

    // Canonical row-major strides for the destination, computed from src.shape.
    // We compute it directly (rather than calling RayzorTensor::compute_strides
    // on a borrowed slice) so we can also size the destination buffer in the
    // same pass and keep this function ABI-shape-only.
    let src_shape_slice: &[usize] = if ndim == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(s.shape, ndim)
    };
    let canonical_strides: Vec<usize> = RayzorTensor::compute_strides(src_shape_slice);

    let data_bytes: usize = s.numel * elem_size;

    // Allocate fresh data buffer.
    let data = malloc(if data_bytes > 0 { data_bytes } else { 1 });
    if data.is_null() {
        return 0;
    }

    // Copy body: contiguous fast-path = memcpy, otherwise strided gather.
    if data_bytes > 0 && !s.data.is_null() {
        if s.is_contiguous() {
            std::ptr::copy_nonoverlapping(s.data, data, data_bytes);
        } else {
            // Strided gather: walk src by its strides, write contiguously
            // into the dest buffer. Dtype-agnostic byte copy across the
            // plain RayzorTensor dtypes (F32/F16/BF16/I32/I64/U8); quantised
            // tensors live in RayzorQTensor and never reach this path.
            let src_strides = std::slice::from_raw_parts(s.strides, ndim);
            let mut idx = vec![0usize; ndim];
            for linear in 0..s.numel {
                // Source byte offset = Σᵢ idx[i] * src_strides[i] * elem_size.
                let mut src_elem_off: usize = 0;
                for axis in 0..ndim {
                    src_elem_off += idx[axis] * src_strides[axis];
                }
                std::ptr::copy_nonoverlapping(
                    s.data.add(src_elem_off * elem_size),
                    data.add(linear * elem_size),
                    elem_size,
                );
                // Increment multi-index (rightmost-axis varies fastest, so
                // dest writes stay sequential).
                for axis in (0..ndim).rev() {
                    idx[axis] += 1;
                    if idx[axis] < src_shape_slice[axis] {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
    }

    // Allocate fresh shape array and copy from src.
    let shape_bytes = ndim * std::mem::size_of::<usize>();
    let shape_ptr = malloc(if shape_bytes > 0 { shape_bytes } else { 1 }) as *mut usize;
    if shape_ptr.is_null() {
        free(data);
        return 0;
    }
    if ndim > 0 && !s.shape.is_null() {
        std::ptr::copy_nonoverlapping(s.shape, shape_ptr, ndim);
    }

    // Allocate fresh strides array seeded with the canonical row-major
    // strides we computed above — NOT inherited from src.
    let strides_ptr = malloc(if shape_bytes > 0 { shape_bytes } else { 1 }) as *mut usize;
    if strides_ptr.is_null() {
        free(data);
        free(shape_ptr as *mut u8);
        return 0;
    }
    if ndim > 0 {
        std::ptr::copy_nonoverlapping(canonical_strides.as_ptr(), strides_ptr, ndim);
    }

    // Allocate the wrapper struct itself.
    let tensor = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if tensor.is_null() {
        free(data);
        free(shape_ptr as *mut u8);
        free(strides_ptr as *mut u8);
        return 0;
    }

    *tensor = RayzorTensor {
        data,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim,
        numel: s.numel,
        dtype: s.dtype,
        owns_data: true,
        device: s.device,
        numa_node: s.numa_node,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
    };

    tensor as i64
}

/// tensor.free() -> void
///
/// Pool-routing semantics:
///
/// - View wrappers (`owns_data == false`) — `reshape`-contiguous, `permute`,
///   `slice`, `transpose`, `transpose_last2` — MUST bypass the pool because
///   their `data` aliases a parent tensor. Pooling them would later hand
///   that alias back as an owning allocation and corrupt the parent on
///   first write. These take the direct free path below: shape/strides
///   wrapper bytes get released, `data` is left alone (the parent owns
///   it).
/// - Owning wrappers (`owns_data == true`) are pushed into
///   `tensor_pool::global()` keyed on `(dtype, shape)`. The pool decides
///   pool-vs-evict; on eviction it invokes `tensor_pool_freer` which runs
///   the same physical release this function would have run inline.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_free(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::TENSOR_FREE);
    TENSOR_FREE_INVOCATIONS.fetch_add(1, MemOrdering::Relaxed);
    let t = &*(tensor_ptr as *const RayzorTensor);

    // Phase 1 ARC: decrement first. Only the thread that drops the count
    // from 1 → 0 proceeds to actually release storage. AcqRel pairs with
    // the Relaxed increments in `rayzor_tensor_arc_clone` / view producers:
    // the Release half of the final dec publishes all prior writes through
    // this wrapper; the Acquire half (only observed by the dec-to-zero
    // thread, which is the sole survivor) prevents reordering of the
    // free below.
    let prev = t.refcount.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    if prev != 1 {
        TENSOR_FREE_REFCOUNT_NONZERO.fetch_add(1, MemOrdering::Relaxed);
        // Other handles still alive. Nothing to do.
        return;
    }

    // We are the sole owner. Snapshot the parent handle (if any) before
    // we tear down our own wrapper — we'll decrement the parent's refcount
    // AFTER releasing our own storage so a recursive view-of-view chain
    // unwinds depth-first.
    let parent = t.parent;
    let owns_data = t.owns_data;

    if !owns_data {
        // Views: drop wrapper + shape/strides; leave data alone — the
        // parent (whose refcount we hold) still owns it.
        if !t.shape.is_null() {
            free(t.shape as *mut u8);
        }
        if !t.strides.is_null() {
            free(t.strides as *mut u8);
        }
        free(tensor_ptr as *mut u8);
        // Drop our reference on the parent. May cascade-free if we were
        // the last view.
        if !parent.is_null() {
            rayzor_tensor_free(parent as i64);
        }
        return;
    }

    // Owning tensor: route through the pool. Build a PooledEntry and let
    // `tensor_pool::global().push()` decide whether to park or free.
    let shape_slice = std::slice::from_raw_parts(t.shape, t.ndim);
    let key = PoolKey::from_shape(t.dtype, shape_slice);
    let alloc_bytes = pool_alloc_bytes(shape_slice, t.dtype);
    let entry = PooledEntry {
        ptr: tensor_ptr as *mut u8,
        shape: ShapeBuf::from_slice(shape_slice),
        alloc_bytes,
        qtensor_meta_ptr: std::ptr::null_mut(),
        qtensor_meta_bytes: 0,
    };
    tensor_pool::global().push(key, entry, tensor_pool_freer);
    // Owning tensors have `parent == null` by construction, so no parent
    // decrement is needed here. (Defensive sanity: if parent ever drifts
    // non-null on an owning tensor, the dec below catches it.)
    if !parent.is_null() {
        rayzor_tensor_free(parent as i64);
    }
}

/// Drain every parked tensor in the global pool. Exposed for end-of-test
/// isolation and benchmark cleanup. Routes the pool's parked entries through
/// the canonical `tensor_pool_freer` so the underlying mallocs are returned
/// to the system allocator.
///
/// Safe to call from any context; no-op when the pool has not been
/// initialised.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_pool_reset() {
    tensor_pool::global().drain(tensor_pool_freer);
}

// ============================================================================
// Helpers
// ============================================================================

/// Read shape from a Haxe Array<Int> data pointer.
/// The pointer points to the raw i64 data of the array.
unsafe fn read_shape(ptr: i64, ndim: usize) -> Vec<usize> {
    if ptr == 0 || ndim == 0 {
        return vec![];
    }
    let data = ptr as *const i64;
    (0..ndim).map(|i| *data.add(i) as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        unsafe {
            let shape = [2usize, 3usize];
            let t = alloc_tensor(&shape, DTYPE_F32, Some(0.0));
            assert!(t != 0);
            let tensor = &*(t as *const RayzorTensor);
            assert_eq!(tensor.ndim, 2);
            assert_eq!(tensor.numel, 6);
            assert_eq!(tensor.dtype, DTYPE_F32);

            // All zeros
            let data = tensor.data as *const f32;
            for i in 0..6 {
                assert_eq!(*data.add(i), 0.0);
            }

            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn test_tensor_ones_and_sum() {
        unsafe {
            let shape = [2usize, 3usize];
            let t = alloc_tensor(&shape, DTYPE_F32, Some(1.0));
            assert!(t != 0);
            let sum = rayzor_tensor_sum(t);
            assert!((sum - 6.0).abs() < 1e-6);
            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn test_tensor_add() {
        unsafe {
            let shape = [3usize];
            let a = alloc_tensor(&shape, DTYPE_F32, Some(2.0));
            let b = alloc_tensor(&shape, DTYPE_F32, Some(3.0));
            let c = rayzor_tensor_add(a, b);
            assert!(c != 0);
            let sum = rayzor_tensor_sum(c);
            assert!((sum - 15.0).abs() < 1e-6); // (2+3)*3 = 15
            rayzor_tensor_free(a);
            rayzor_tensor_free(b);
            rayzor_tensor_free(c);
        }
    }

    #[test]
    fn test_tensor_matmul() {
        unsafe {
            // [2,2] identity matmul [2,2] ones = [2,2] with row sums = 2
            let ident = alloc_tensor(&[2, 2], DTYPE_F32, Some(0.0));
            let ones = alloc_tensor(&[2, 2], DTYPE_F32, Some(1.0));

            // Set identity
            let id = &*(ident as *const RayzorTensor);
            let d = id.data as *mut f32;
            *d.add(0) = 1.0; // [0,0]
            *d.add(3) = 1.0; // [1,1]

            let result = rayzor_tensor_matmul(ident, ones);
            assert!(result != 0);
            let sum = rayzor_tensor_sum(result);
            assert!((sum - 4.0).abs() < 1e-6); // I * ones = ones, sum = 4

            rayzor_tensor_free(ident);
            rayzor_tensor_free(ones);
            rayzor_tensor_free(result);
        }
    }

    #[test]
    fn from_bytes_f16_widens_to_f32() {
        // f16 0x3C00=1.0, 0x4000=2.0, 0xBC00=-1.0, 0x3800=0.5
        let mut buf: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40, 0x00, 0xBC, 0x00, 0x38];
        let bytes =
            crate::haxe_sys::HaxeBytes::new_malloc(buf.as_mut_ptr(), buf.len(), buf.capacity());
        let mut shape: Vec<i64> = vec![4];
        unsafe {
            let t = rayzor_tensor_from_bytes_f16(
                &bytes as *const _ as i64,
                shape.as_mut_ptr() as i64,
                1,
            );
            assert!(t != 0);
            let tensor = &*(t as *const RayzorTensor);
            assert_eq!(tensor.dtype, DTYPE_F32);
            let d = tensor.data as *const f32;
            assert!((*d.add(0) - 1.0).abs() < 1e-3);
            assert!((*d.add(1) - 2.0).abs() < 1e-3);
            assert!((*d.add(2) - -1.0).abs() < 1e-3);
            assert!((*d.add(3) - 0.5).abs() < 1e-3);
            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn from_bytes_q8_0_dequant() {
        // One 34-byte Q8_0 block: f16(0.5) scale + 32 i8 weights.
        // Expected outputs: weights * 0.5.
        let mut buf = vec![0u8; 34];
        buf[0] = 0x00;
        buf[1] = 0x38; // f16 0.5
        buf[2] = 2; // → 1.0
        buf[3] = 0xFE; // -2 → -1.0
        buf[4] = 0; // 0.0
        buf[5] = 4; // 2.0
        let bytes =
            crate::haxe_sys::HaxeBytes::new_malloc(buf.as_mut_ptr(), buf.len(), buf.capacity());
        let mut shape: Vec<i64> = vec![32];
        unsafe {
            let t = rayzor_tensor_from_bytes_q8_0(
                &bytes as *const _ as i64,
                shape.as_mut_ptr() as i64,
                1,
            );
            assert!(t != 0);
            let tensor = &*(t as *const RayzorTensor);
            assert_eq!(tensor.dtype, DTYPE_F32);
            let d = tensor.data as *const f32;
            assert!((*d.add(0) - 1.0).abs() < 1e-3);
            assert!((*d.add(1) - -1.0).abs() < 1e-3);
            assert!((*d.add(2) - 0.0).abs() < 1e-3);
            assert!((*d.add(3) - 2.0).abs() < 1e-3);
            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn add_into_f32_contiguous_fast_path() {
        unsafe {
            let shape = [4usize];
            let dest = alloc_tensor(&shape, DTYPE_F32, None);
            let src = alloc_tensor(&shape, DTYPE_F32, None);
            assert!(dest != 0 && src != 0);

            let d = &*(dest as *const RayzorTensor);
            let s = &*(src as *const RayzorTensor);
            let d_data = d.data as *mut f32;
            let s_data = s.data as *mut f32;
            *d_data.add(0) = 1.0;
            *d_data.add(1) = 2.0;
            *d_data.add(2) = 3.0;
            *d_data.add(3) = 4.0;
            *s_data.add(0) = 10.0;
            *s_data.add(1) = 20.0;
            *s_data.add(2) = 30.0;
            *s_data.add(3) = 40.0;

            rayzor_tensor_add_into(dest, src);

            // dest mutated to [11, 22, 33, 44]
            assert!((*d_data.add(0) - 11.0).abs() < 1e-6);
            assert!((*d_data.add(1) - 22.0).abs() < 1e-6);
            assert!((*d_data.add(2) - 33.0).abs() < 1e-6);
            assert!((*d_data.add(3) - 44.0).abs() < 1e-6);

            // src untouched at [10, 20, 30, 40]
            assert!((*s_data.add(0) - 10.0).abs() < 1e-6);
            assert!((*s_data.add(1) - 20.0).abs() < 1e-6);
            assert!((*s_data.add(2) - 30.0).abs() < 1e-6);
            assert!((*s_data.add(3) - 40.0).abs() < 1e-6);

            rayzor_tensor_free(dest);
            rayzor_tensor_free(src);
        }
    }

    #[test]
    fn add_into_f32_tail_lanes_correct() {
        // numel = 7 → not a multiple of 4-lane SIMD; verifies the
        // tail-element scalar fallback in add_slice.
        unsafe {
            let shape = [7usize];
            let dest = alloc_tensor(&shape, DTYPE_F32, Some(1.0));
            let src = alloc_tensor(&shape, DTYPE_F32, Some(2.5));
            rayzor_tensor_add_into(dest, src);
            let d = &*(dest as *const RayzorTensor);
            let d_data = d.data as *const f32;
            for i in 0..7 {
                assert!((*d_data.add(i) - 3.5).abs() < 1e-6, "elem {} wrong", i);
            }
            rayzor_tensor_free(dest);
            rayzor_tensor_free(src);
        }
    }

    #[test]
    fn add_into_is_noop_when_numel_zero() {
        // 0-numel shape is a no-op; the function must return cleanly
        // without touching the (1-byte sentinel) backing buffer.
        unsafe {
            let shape = [0usize, 3];
            let dest = alloc_tensor(&shape, DTYPE_F32, None);
            let src = alloc_tensor(&shape, DTYPE_F32, None);
            assert!(dest != 0 && src != 0);
            rayzor_tensor_add_into(dest, src);
            rayzor_tensor_free(dest);
            rayzor_tensor_free(src);
        }
    }

    #[test]
    fn from_bytes_f16_rejects_short_buffer() {
        let mut buf = vec![0u8; 2]; // only 1 element worth
        let bytes =
            crate::haxe_sys::HaxeBytes::new_malloc(buf.as_mut_ptr(), buf.len(), buf.capacity());
        let mut shape: Vec<i64> = vec![4]; // asks for 4 elements (8 bytes)
        unsafe {
            assert_eq!(
                rayzor_tensor_from_bytes_f16(
                    &bytes as *const _ as i64,
                    shape.as_mut_ptr() as i64,
                    1,
                ),
                0
            );
        }
    }

    /// Clone of a permuted view must compact to canonical row-major strides,
    /// own a `numel * elem_size` byte buffer, and preserve element values
    /// under the permuted indexing. Regression for the clone-of-view memory-
    /// safety footgun (bugs_clone_view_passthrough_invariant.md).
    #[test]
    fn tensor_clone_of_permuted_view_compacts_to_contiguous() {
        unsafe {
            // Build a 2x3 F32 tensor with values 0..6, laid out row-major.
            let src_shape = [2usize, 3usize];
            let src = alloc_tensor(&src_shape, DTYPE_F32, None);
            assert!(src != 0);
            let src_ref = &*(src as *const RayzorTensor);
            let src_data = src_ref.data as *mut f32;
            for i in 0..6 {
                *src_data.add(i) = i as f32;
            }

            // Permute([1, 0]) → view with shape [3, 2] and non-contiguous
            // strides [1, 3] (rows of the source become columns of the view).
            let axes: [i64; 2] = [1, 0];
            let view = rayzor_tensor_permute(src, axes.as_ptr() as i64, 2);
            assert!(view != 0);
            let view_ref = &*(view as *const RayzorTensor);
            assert!(!view_ref.owns_data, "permute must produce a view");
            assert!(
                !view_ref.is_contiguous(),
                "permuted view must NOT be contiguous"
            );
            let view_strides = std::slice::from_raw_parts(view_ref.strides, view_ref.ndim);
            assert_eq!(view_strides, &[1usize, 3usize]);

            // Deep-clone the view. Result must be owning, contiguous, with
            // canonical row-major strides for shape [3, 2] = [2, 1], and a
            // data buffer of exactly `numel * sizeof(f32)` = 24 bytes.
            // (Phase 1: `rayzor_tensor_clone` is now Arc-increment; this
            // compact-to-contiguous invariant lives on `rayzor_tensor_deep_clone`.)
            let cloned = rayzor_tensor_deep_clone(view);
            assert!(cloned != 0);
            let cloned_ref = &*(cloned as *const RayzorTensor);
            assert!(cloned_ref.owns_data, "deep_clone must own its data");
            assert!(
                cloned_ref.is_contiguous(),
                "clone of a view must be contiguous"
            );
            assert_eq!(cloned_ref.ndim, 2);
            assert_eq!(cloned_ref.numel, 6);
            let cloned_shape = std::slice::from_raw_parts(cloned_ref.shape, 2);
            let cloned_strides = std::slice::from_raw_parts(cloned_ref.strides, 2);
            assert_eq!(cloned_shape, &[3usize, 2usize]);
            assert_eq!(
                cloned_strides,
                &[2usize, 1usize],
                "clone must seed canonical row-major strides, not inherit view strides"
            );

            // Element-by-element equality between view and clone under
            // permuted indexing. View element [i, j] = src[j, i] = j*3 + i.
            // Clone reads via canonical strides (i*2 + j).
            let clone_data = cloned_ref.data as *const f32;
            for i in 0..3 {
                for j in 0..2 {
                    let expected = (j * 3 + i) as f32;
                    // Read via view's offset() helper (uses view strides).
                    let view_off = view_ref.offset(&[i, j]);
                    let view_val = *(view_ref.data as *const f32).add(view_off);
                    assert_eq!(view_val, expected, "view[{},{}] wrong", i, j);
                    // Read via clone's contiguous row-major layout.
                    let clone_val = *clone_data.add(i * 2 + j);
                    assert_eq!(clone_val, expected, "clone[{},{}] wrong", i, j);
                }
            }

            // Independence: mutating the clone must not touch the source.
            let clone_mut = cloned_ref.data as *mut f32;
            *clone_mut = 999.0;
            assert_eq!(*src_data, 0.0, "source aliased through clone");

            rayzor_tensor_free(cloned);
            rayzor_tensor_free(view);
            rayzor_tensor_free(src);
        }
    }

    /// Deep-clone of an already-contiguous tensor must still produce a fresh
    /// owning contiguous tensor (memcpy fast-path), independent from the source.
    /// (Phase 1: the disjoint-storage path moved from `rayzor_tensor_clone` to
    /// `rayzor_tensor_deep_clone`.)
    #[test]
    fn tensor_deep_clone_of_contiguous_is_independent() {
        unsafe {
            let shape = [4usize];
            let src = alloc_tensor(&shape, DTYPE_F32, Some(7.5));
            assert!(src != 0);
            let cloned = rayzor_tensor_deep_clone(src);
            assert!(cloned != 0);
            let cloned_ref = &*(cloned as *const RayzorTensor);
            assert!(cloned_ref.owns_data);
            assert!(cloned_ref.is_contiguous());
            assert_eq!(cloned_ref.numel, 4);
            let cd = cloned_ref.data as *mut f32;
            for i in 0..4 {
                assert_eq!(*cd.add(i), 7.5);
            }
            // Mutate clone; source untouched.
            *cd = -1.0;
            let src_ref = &*(src as *const RayzorTensor);
            let sd = src_ref.data as *const f32;
            assert_eq!(*sd, 7.5);
            rayzor_tensor_free(cloned);
            rayzor_tensor_free(src);
        }
    }

    // ========================================================================
    // Phase 1 ARC refcount tests
    // ========================================================================

    /// `rayzor_tensor_arc_clone` returns the SAME pointer (i64-handle ABI
    /// preservation), and the underlying buffer + wrapper stay alive after a
    /// matching `rayzor_tensor_free` because the refcount only went 2→1.
    #[test]
    fn arc_clone_then_free_leaves_original_alive() {
        unsafe {
            let shape = [4usize];
            let t = alloc_tensor(&shape, DTYPE_F32, Some(11.0));
            assert!(t != 0);

            let cloned = rayzor_tensor_arc_clone(t);
            assert_eq!(cloned, t, "arc_clone must return the same handle");

            // Refcount should be 2 (initial 1 + clone).
            let tref = &*(t as *const RayzorTensor);
            assert_eq!(tref.refcount.load(std::sync::atomic::Ordering::Relaxed), 2);

            // First free: only decrements; storage must stay alive.
            rayzor_tensor_free(cloned);

            // Refcount should be 1 now, data still readable.
            let tref2 = &*(t as *const RayzorTensor);
            assert_eq!(tref2.refcount.load(std::sync::atomic::Ordering::Relaxed), 1);
            let d = tref2.data as *const f32;
            for i in 0..4 {
                assert_eq!(*d.add(i), 11.0, "data corrupted after first free");
            }

            // Second free: dec-to-zero, real release.
            rayzor_tensor_free(t);
        }
    }

    /// Double arc_clone + double free should net to a single physical free
    /// (refcount 1→2→3→2→1→0). The pool / direct-free path runs exactly once
    /// on the final dec-to-zero; storage must NOT be released earlier.
    #[test]
    fn double_clone_then_double_free_deallocates_exactly_once() {
        unsafe {
            let shape = [3usize];
            let t = alloc_tensor(&shape, DTYPE_F32, Some(5.0));
            assert!(t != 0);

            let a = rayzor_tensor_arc_clone(t);
            let b = rayzor_tensor_arc_clone(t);
            assert_eq!(a, t);
            assert_eq!(b, t);

            let tref = &*(t as *const RayzorTensor);
            assert_eq!(tref.refcount.load(std::sync::atomic::Ordering::Relaxed), 3);

            rayzor_tensor_free(a);
            rayzor_tensor_free(b);

            // Refcount should be 1; storage still alive.
            let tref2 = &*(t as *const RayzorTensor);
            assert_eq!(tref2.refcount.load(std::sync::atomic::Ordering::Relaxed), 1);
            let d = tref2.data as *const f32;
            for i in 0..3 {
                assert_eq!(*d.add(i), 5.0);
            }

            // Final free actually releases.
            rayzor_tensor_free(t);
        }
    }

    /// Producing a view (permute) bumps the parent's refcount so the parent
    /// data buffer stays alive across `parent.free()` until the view is also
    /// freed. After both frees the buffer is reclaimed.
    #[test]
    fn view_clone_increments_parent_ref() {
        unsafe {
            let shape = [2usize, 3usize];
            let src = alloc_tensor(&shape, DTYPE_F32, None);
            assert!(src != 0);
            let src_data = (&*(src as *const RayzorTensor)).data as *mut f32;
            for i in 0..6 {
                *src_data.add(i) = i as f32;
            }

            // Parent starts at refcount 1.
            let src_ref0 = &*(src as *const RayzorTensor);
            assert_eq!(
                src_ref0.refcount.load(std::sync::atomic::Ordering::Relaxed),
                1
            );

            // Make a permuted view. View should be a fresh wrapper, parent
            // refcount bumped to 2.
            let axes: [i64; 2] = [1, 0];
            let view = rayzor_tensor_permute(src, axes.as_ptr() as i64, 2);
            assert!(view != 0);
            assert_ne!(view, src);

            let src_ref1 = &*(src as *const RayzorTensor);
            assert_eq!(
                src_ref1.refcount.load(std::sync::atomic::Ordering::Relaxed),
                2,
                "view producer must bump parent refcount"
            );
            let view_ref = &*(view as *const RayzorTensor);
            assert!(!view_ref.owns_data, "permute must be a view");
            assert_eq!(view_ref.parent, src as *mut RayzorTensor);

            // Free the parent handle first. Data must still be alive because
            // the view holds a refcount.
            rayzor_tensor_free(src);

            // src wrapper should still be readable (parent refcount = 1).
            let src_ref2 = &*(src as *const RayzorTensor);
            assert_eq!(
                src_ref2.refcount.load(std::sync::atomic::Ordering::Relaxed),
                1
            );
            // View still points at live data — the test is that this read
            // doesn't segfault, i.e. the parent's data buffer is still alive
            // even though `src` has been freed. Read element 0 (always src[0]
            // regardless of stride pattern); a UAF here would either segfault
            // or return garbage instead of the seeded value 0.0.
            let view_data = (&*(view as *const RayzorTensor)).data as *const f32;
            assert_eq!(
                *view_data.add(0),
                0.0,
                "view read after parent free returned wrong value (parent buffer freed early?)"
            );

            // Free the view. Should cascade-free the parent.
            rayzor_tensor_free(view);
        }
    }
}
