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

/// Kernel function name for matmul.
pub fn matmul_fn_name(dtype: u8) -> String {
    format!("rayzor_matmul_{}", dtype_to_wgsl(dtype))
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
}
