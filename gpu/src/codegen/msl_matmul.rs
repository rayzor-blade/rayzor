//! MSL code generation for matrix multiplication.
//!
//! Tiled 16×16 shared-memory matmul: C[M×N] = A[M×K] × B[K×N].
//! Each threadgroup loads a 16×16 tile of A and B into shared memory,
//! computes partial sums, then moves to the next tile along K.
//! This reduces global memory accesses from O(K) per thread to O(K/16).
//!
//! Dispatched as a 2D grid with 16×16 threads per threadgroup.
//! Dimensions (M, K, N) are passed via a constant buffer.

use super::msl::dtype_to_msl;

const TILE_SIZE: usize = 16;

/// Generate MSL source for tiled matrix multiplication.
///
/// Buffers: A (M×K), B (K×N), C (M×N), dims (uint4: M, K, N, 0)
pub fn emit_matmul(dtype: u8) -> String {
    let msl_type = dtype_to_msl(dtype);
    let fn_name = format!("rayzor_matmul_{}", msl_type);
    let ts = TILE_SIZE;

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {fn_name}(
    device const {msl_type}* A [[buffer(0)]],
    device const {msl_type}* B [[buffer(1)]],
    device {msl_type}* C [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {{
    const uint TILE = {ts};
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    // Global row/col this thread computes
    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    // Shared memory tiles
    threadgroup {msl_type} As[{ts}][{ts}];
    threadgroup {msl_type} Bs[{ts}][{ts}];

    {msl_type} sum = 0;

    // Slide the tile window along K dimension
    uint numTiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; t++) {{
        // Load A tile: A[row, t*TILE + tid.x]
        uint a_col = t * TILE + tid.x;
        if (row < M && a_col < K) {{
            As[tid.y][tid.x] = A[row * K + a_col];
        }} else {{
            As[tid.y][tid.x] = 0;
        }}

        // Load B tile: B[t*TILE + tid.y, col]
        uint b_row = t * TILE + tid.y;
        if (b_row < K && col < N) {{
            Bs[tid.y][tid.x] = B[b_row * N + col];
        }} else {{
            Bs[tid.y][tid.x] = 0;
        }}

        // Sync to ensure tile is fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate dot product of row from As and col from Bs
        for (uint i = 0; i < TILE; i++) {{
            sum = fma(As[tid.y][i], Bs[i][tid.x], sum);
        }}

        // Sync before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write result
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#
    )
}

/// Kernel function name for matmul.
pub fn matmul_fn_name(dtype: u8) -> String {
    format!("rayzor_matmul_{}", dtype_to_msl(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32() {
        let src = emit_matmul(crate::buffer::DTYPE_F32);
        assert!(src.contains("kernel void rayzor_matmul_float"));
        assert!(src.contains("device const float* A"));
        assert!(src.contains("device const float* B"));
        assert!(src.contains("device float* C"));
        assert!(src.contains("constant uint4& dims"));
        assert!(src.contains("threadgroup float As[16][16]"));
        assert!(src.contains("threadgroup float Bs[16][16]"));
        assert!(src.contains("threadgroup_barrier"));
        assert!(src.contains("fma("));
    }
}
