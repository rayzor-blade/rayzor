//! WGSL code generation for matrix multiplication.
//!
//! Tiled 16×16 shared-memory matmul: C[M×N] = A[M×K] × B[K×N].
//! Each workgroup loads a 16×16 tile of A and B into workgroup memory,
//! computes partial sums, then moves to the next tile along K.
//!
//! Dispatched as a 2D grid with 16×16 threads per workgroup.
//! Dimensions (M, K, N) are passed via a uniform buffer.

use super::wgsl::dtype_to_wgsl;

const TILE_SIZE: usize = 16;

/// Generate WGSL source for tiled matrix multiplication.
///
/// Buffers: A (M×K), B (K×N), C (M×N), dims (vec4<u32>: M, K, N, 0)
pub fn emit_matmul(dtype: u8) -> String {
    let wgsl_type = dtype_to_wgsl(dtype);
    let fn_name = format!("rayzor_matmul_{}", wgsl_type);
    let ts = TILE_SIZE;

    format!(
        r#"@group(0) @binding(0) var<storage, read> A: array<{wgsl_type}>;
@group(0) @binding(1) var<storage, read> B: array<{wgsl_type}>;
@group(0) @binding(2) var<storage, read_write> C: array<{wgsl_type}>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

const TILE: u32 = {ts}u;

var<workgroup> As: array<array<{wgsl_type}, {ts}>, {ts}>;
var<workgroup> Bs: array<array<{wgsl_type}, {ts}>, {ts}>;

@compute @workgroup_size({ts}, {ts})
fn {fn_name}(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>
) {{
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    let row = tgid.y * TILE + tid.y;
    let col = tgid.x * TILE + tid.x;

    var sum = {wgsl_type}(0);

    let numTiles = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < numTiles; t = t + 1u) {{
        // Load A tile
        let a_col = t * TILE + tid.x;
        if (row < M && a_col < K) {{
            As[tid.y][tid.x] = A[row * K + a_col];
        }} else {{
            As[tid.y][tid.x] = {wgsl_type}(0);
        }}

        // Load B tile
        let b_row = t * TILE + tid.y;
        if (b_row < K && col < N) {{
            Bs[tid.y][tid.x] = B[b_row * N + col];
        }} else {{
            Bs[tid.y][tid.x] = {wgsl_type}(0);
        }}

        workgroupBarrier();

        // Accumulate
        for (var i = 0u; i < TILE; i = i + 1u) {{
            sum = fma(As[tid.y][i], Bs[i][tid.x], sum);
        }}

        workgroupBarrier();
    }}

    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#
    )
}

// ---------------------------------------------------------------------------
// Register-tiled matmul
// ---------------------------------------------------------------------------
//
// The 16x16 kernel above computes ONE output per thread: each k-step costs two
// shared-memory loads for a single FMA, so the ALUs starve on SLM traffic and
// the kernel tops out near a tenth of peak (measured 128 GF/s = 7% of an Iris
// Xe 80EU's ~1.79 TFLOPS).
//
// Here each thread owns a TM x TN block of C held in registers. A k-step loads
// TM + TN values from shared memory and issues TM * TN FMAs, taking the
// arithmetic-to-load ratio from 1:2 to 16:8 at TM=TN=4 — an 8x improvement in
// what the ALUs get per byte of SLM.
//
// Workgroup: 16x16 = 256 threads, each producing 4x4 outputs -> a 64x64 tile.
// Shared per workgroup: 64*16 + 16*64 floats = 8 KB, well inside the 64 KB SLM.

// Swept on Iris Xe (Alder Lake-P GT2) against a 653x4096x14336 prefill shape,
// 3 runs each, GF/s:
//   64x128 BK16 4x8  382 380 380   <- default
//   64x64  BK32 4x4  385 377 377   (tied within noise)
//   128x64 BK16 8x4  365 367 363
//   64x64  BK16 4x4  325           (first guess, -15%)
//   128x128 BK16 8x8  24           <- SPILLS: 64 accumulators per thread
// 8x8 per thread collapses 15x on register pressure, the same wall that made
// the CPU-side 4->8 tile widening lose. Keep TM*TN <= 32.
const BM: usize = 64; // output rows per workgroup
const BN: usize = 128; // output cols per workgroup
const BK: usize = 16; // k-slice depth
const TM: usize = 4; // rows per thread
const TN: usize = 8; // cols per thread

/// Generate WGSL for a register-tiled matmul. Same buffer layout and entry
/// point name as [`emit_matmul`], so it is a drop-in replacement.
pub fn emit_matmul_tiled(dtype: u8) -> String {
    let wgsl_type = dtype_to_wgsl(dtype);
    let fn_name = format!("rayzor_matmul_{}", wgsl_type);
    let (bm, bn, bk, tm, tn) = tile_config();
    let threads = (bm / tm) * (bn / tn);

    format!(
        r#"@group(0) @binding(0) var<storage, read> A: array<{wgsl_type}>;
@group(0) @binding(1) var<storage, read> B: array<{wgsl_type}>;
@group(0) @binding(2) var<storage, read_write> C: array<{wgsl_type}>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const NTHREADS: u32 = {threads}u;

var<workgroup> As: array<{wgsl_type}, {bm}u * {bk}u>;
var<workgroup> Bs: array<{wgsl_type}, {bk}u * {bn}u>;

@compute @workgroup_size({bn}u / {tn}u, {bm}u / {tm}u)
fn {fn_name}(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>
) {{
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    let blockRow = tgid.y * BM;
    let blockCol = tgid.x * BN;
    let lin = tid.y * (BN / TN) + tid.x;

    // Per-thread output block, held in registers.
    var acc: array<{wgsl_type}, {tm}u * {tn}u>;
    for (var i = 0u; i < TM * TN; i = i + 1u) {{
        acc[i] = {wgsl_type}(0);
    }}

    let numTiles = (K + BK - 1u) / BK;
    for (var t = 0u; t < numTiles; t = t + 1u) {{
        let kBase = t * BK;

        // Cooperative load: BM*BK and BK*BN elements across NTHREADS threads.
        for (var i = lin; i < BM * BK; i = i + NTHREADS) {{
            let r = i / BK;
            let c = i % BK;
            let gr = blockRow + r;
            let gc = kBase + c;
            if (gr < M && gc < K) {{
                As[i] = A[gr * K + gc];
            }} else {{
                As[i] = {wgsl_type}(0);
            }}
        }}
        for (var i = lin; i < BK * BN; i = i + NTHREADS) {{
            let r = i / BN;
            let c = i % BN;
            let gr = kBase + r;
            let gc = blockCol + c;
            if (gr < K && gc < N) {{
                Bs[i] = B[gr * N + gc];
            }} else {{
                Bs[i] = {wgsl_type}(0);
            }}
        }}

        workgroupBarrier();

        // TM+TN shared loads feed TM*TN FMAs.
        for (var k = 0u; k < BK; k = k + 1u) {{
            var a: array<{wgsl_type}, {tm}u>;
            var b: array<{wgsl_type}, {tn}u>;
            for (var i = 0u; i < TM; i = i + 1u) {{
                a[i] = As[(tid.y * TM + i) * BK + k];
            }}
            for (var j = 0u; j < TN; j = j + 1u) {{
                b[j] = Bs[k * BN + tid.x * TN + j];
            }}
            for (var i = 0u; i < TM; i = i + 1u) {{
                for (var j = 0u; j < TN; j = j + 1u) {{
                    acc[i * TN + j] = fma(a[i], b[j], acc[i * TN + j]);
                }}
            }}
        }}

        workgroupBarrier();
    }}

    for (var i = 0u; i < TM; i = i + 1u) {{
        let gr = blockRow + tid.y * TM + i;
        if (gr >= M) {{ continue; }}
        for (var j = 0u; j < TN; j = j + 1u) {{
            let gc = blockCol + tid.x * TN + j;
            if (gc < N) {{
                C[gr * N + gc] = acc[i * TN + j];
            }}
        }}
    }}
}}
"#
    )
}

/// Tile shape, overridable for sweeps via RZG_BM/BN/BK/TM/TN.
///
/// Constraints the caller must respect: (BM/TM)*(BN/TN) is the workgroup size
/// and must not exceed the device max (256 on Iris Xe); BM*BK + BK*BN floats
/// must fit shared memory; BM%TM == 0 and BN%TN == 0.
pub fn tile_config() -> (usize, usize, usize, usize, usize) {
    fn env(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(default)
    }
    (
        env("RZG_BM", BM),
        env("RZG_BN", BN),
        env("RZG_BK", BK),
        env("RZG_TM", TM),
        env("RZG_TN", TN),
    )
}

/// Output-tile height/width one workgroup covers, so the dispatcher can size
/// its grid to match whatever `tile_config` returned.
pub fn tiled_bm() -> u32 {
    tile_config().0 as u32
}
pub fn tiled_bn() -> u32 {
    tile_config().1 as u32
}

/// `RZG_MATMUL=simple` falls back to the original one-output-per-thread kernel
/// so the two can be A/B'd in a single binary.
pub fn use_tiled_matmul() -> bool {
    !matches!(
        std::env::var("RZG_MATMUL").as_deref(),
        Ok("simple") | Ok("SIMPLE")
    )
}

/// Kernel function name for matmul.
pub fn matmul_fn_name(dtype: u8) -> String {
    format!("rayzor_matmul_{}", dtype_to_wgsl(dtype))
}

/// Generate WGSL source for batched matrix multiplication.
///
/// C[b,m,n] = A[b,m,k] × B[b,k,n] for each batch b.
/// dims = vec4<u32>(M, K, N, B)
/// Dispatched as 3D: (ceil(N/16), ceil(M/16), B) workgroups.
pub fn emit_batch_matmul(dtype: u8) -> String {
    let wgsl_type = dtype_to_wgsl(dtype);
    let fn_name = format!("rayzor_batch_matmul_{}", wgsl_type);
    let ts = TILE_SIZE;

    format!(
        r#"@group(0) @binding(0) var<storage, read> A: array<{wgsl_type}>;
@group(0) @binding(1) var<storage, read> B: array<{wgsl_type}>;
@group(0) @binding(2) var<storage, read_write> C: array<{wgsl_type}>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

const TILE: u32 = {ts}u;

var<workgroup> As: array<array<{wgsl_type}, {ts}>, {ts}>;
var<workgroup> Bs: array<array<{wgsl_type}, {ts}>, {ts}>;

@compute @workgroup_size({ts}, {ts})
fn {fn_name}(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>
) {{
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;
    let batch = tgid.z;

    let row = tgid.y * TILE + tid.y;
    let col = tgid.x * TILE + tid.x;

    let a_offset = batch * M * K;
    let b_offset = batch * K * N;
    let c_offset = batch * M * N;

    var sum = {wgsl_type}(0);

    let numTiles = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < numTiles; t = t + 1u) {{
        let a_col = t * TILE + tid.x;
        if (row < M && a_col < K) {{
            As[tid.y][tid.x] = A[a_offset + row * K + a_col];
        }} else {{
            As[tid.y][tid.x] = {wgsl_type}(0);
        }}

        let b_row = t * TILE + tid.y;
        if (b_row < K && col < N) {{
            Bs[tid.y][tid.x] = B[b_offset + b_row * N + col];
        }} else {{
            Bs[tid.y][tid.x] = {wgsl_type}(0);
        }}

        workgroupBarrier();

        for (var i = 0u; i < TILE; i = i + 1u) {{
            sum = fma(As[tid.y][i], Bs[i][tid.x], sum);
        }}

        workgroupBarrier();
    }}

    if (row < M && col < N) {{
        C[c_offset + row * N + col] = sum;
    }}
}}
"#
    )
}

/// Kernel function name for batch matmul.
pub fn batch_matmul_fn_name(dtype: u8) -> String {
    format!("rayzor_batch_matmul_{}", dtype_to_wgsl(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32() {
        let src = emit_matmul(crate::buffer::DTYPE_F32);
        assert!(src.contains("fn rayzor_matmul_f32"));
        assert!(src.contains("var<storage, read> A: array<f32>"));
        assert!(src.contains("var<workgroup> As: array<array<f32, 16>, 16>"));
        assert!(src.contains("var<workgroup> Bs: array<array<f32, 16>, 16>"));
        assert!(src.contains("workgroupBarrier()"));
        assert!(src.contains("fma("));
        assert!(src.contains("@workgroup_size(16, 16)"));
    }

    #[test]
    fn test_batch_matmul_f32() {
        let src = emit_batch_matmul(crate::buffer::DTYPE_F32);
        assert!(src.contains("fn rayzor_batch_matmul_f32"));
        assert!(src.contains("let batch = tgid.z"));
        assert!(src.contains("a_offset"));
        assert!(src.contains("c_offset"));
        assert!(src.contains("workgroupBarrier()"));
    }
}
