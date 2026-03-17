//! CUDA C code generation for matrix multiplication.
//!
//! Tiled 16×16 shared-memory matmul, same algorithm as MSL/WGSL versions.

use super::cuda::dtype_to_cuda;
use crate::buffer;

const TILE_SIZE: usize = 16;

/// Generate CUDA source for tiled matrix multiplication.
pub fn emit_matmul(dtype: u8) -> String {
    let cuda_type = dtype_to_cuda(dtype);
    let fn_name = matmul_fn_name(dtype);
    let ts = TILE_SIZE;

    format!(
        r#"extern "C" __global__ void {fn_name}(
    const {cuda_type}* A,
    const {cuda_type}* B,
    {cuda_type}* C,
    unsigned int M, unsigned int K, unsigned int N
) {{
    const unsigned int TILE = {ts};

    unsigned int row = blockIdx.y * TILE + threadIdx.y;
    unsigned int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ {cuda_type} As[{ts}][{ts}];
    __shared__ {cuda_type} Bs[{ts}][{ts}];

    {cuda_type} sum = 0;

    unsigned int numTiles = (K + TILE - 1) / TILE;
    for (unsigned int t = 0; t < numTiles; t++) {{
        unsigned int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        unsigned int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (unsigned int i = 0; i < TILE; i++)
            sum = fma(As[threadIdx.y][i], Bs[i][threadIdx.x], sum);

        __syncthreads();
    }}

    if (row < M && col < N)
        C[row * N + col] = sum;
}}
"#
    )
}

/// Generate CUDA source for batched matrix multiplication.
pub fn emit_batch_matmul(dtype: u8) -> String {
    let cuda_type = dtype_to_cuda(dtype);
    let fn_name = batch_matmul_fn_name(dtype);
    let ts = TILE_SIZE;

    format!(
        r#"extern "C" __global__ void {fn_name}(
    const {cuda_type}* A,
    const {cuda_type}* B,
    {cuda_type}* C,
    unsigned int M, unsigned int K, unsigned int N, unsigned int batch_count
) {{
    const unsigned int TILE = {ts};
    unsigned int batch = blockIdx.z;

    unsigned int row = blockIdx.y * TILE + threadIdx.y;
    unsigned int col = blockIdx.x * TILE + threadIdx.x;

    unsigned int a_off = batch * M * K;
    unsigned int b_off = batch * K * N;
    unsigned int c_off = batch * M * N;

    __shared__ {cuda_type} As[{ts}][{ts}];
    __shared__ {cuda_type} Bs[{ts}][{ts}];

    {cuda_type} sum = 0;

    unsigned int numTiles = (K + TILE - 1) / TILE;
    for (unsigned int t = 0; t < numTiles; t++) {{
        unsigned int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[a_off + row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        unsigned int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_off + b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (unsigned int i = 0; i < TILE; i++)
            sum = fma(As[threadIdx.y][i], Bs[i][threadIdx.x], sum);

        __syncthreads();
    }}

    if (row < M && col < N)
        C[c_off + row * N + col] = sum;
}}
"#
    )
}

pub fn matmul_fn_name(dtype: u8) -> String {
    format!("rayzor_matmul_{}", dtype_to_cuda(dtype).replace(' ', "_"))
}

pub fn batch_matmul_fn_name(dtype: u8) -> String {
    format!("rayzor_batch_matmul_{}", dtype_to_cuda(dtype).replace(' ', "_"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32() {
        let src = emit_matmul(buffer::DTYPE_F32);
        assert!(src.contains("rayzor_matmul_float"));
        assert!(src.contains("__shared__ float As[16][16]"));
        assert!(src.contains("__syncthreads()"));
        assert!(src.contains("fma("));
    }

    #[test]
    fn test_batch_matmul_f32() {
        let src = emit_batch_matmul(buffer::DTYPE_F32);
        assert!(src.contains("rayzor_batch_matmul_float"));
        assert!(src.contains("blockIdx.z"));
        assert!(src.contains("a_off"));
    }
}
