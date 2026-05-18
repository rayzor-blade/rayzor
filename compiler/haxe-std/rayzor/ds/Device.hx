package rayzor.ds;

/**
 * Where a Tensor lives.
 *
 * Desktop / mobile / browser coverage:
 * - `CPU(node)` — main memory, optionally pinned to a NUMA node. `node = -1`
 *   means "anywhere" (no affinity hint); `node >= 0` is a specific NUMA node
 *   from `rayzor.concurrent.NumaTopology`. On platforms without NUMA
 *   (macOS, single-socket Linux, wasm) any `node >= 0` collapses to node 0.
 * - `Metal` — native Apple GPU path (macOS, iOS). Bypasses wgpu.
 * - `Cuda` — native NVIDIA path (Linux, Windows). Bypasses wgpu. Best perf
 *   on NVIDIA, but only when the CUDA toolkit is installed.
 * - `Vulkan` — Linux / Windows GPU via `wgpu` adapter pinned to Vulkan. The
 *   only GPU path on AMD / Intel desktop GPUs, and a portable fallback for
 *   NVIDIA when CUDA isn't available. Sufficient for ML inference (SPIR-V
 *   compute shaders cover everything WGSL/MSL can express).
 * - `WebGPU` — browser GPU via `wgpu` running in the wasm target. Same
 *   shader source path as Vulkan (the WebGPU adapter under the hood).
 *
 * Buffers on any GPU device live in device memory; readback to CPU requires
 * an explicit `.to(CPU(-1))`.
 *
 * Map to runtime tags (u8) for the FFI: CPU=0, Metal=1, Cuda=2, Vulkan=3, WebGPU=4.
 */
enum Device {
    CPU(node:Int);
    Metal;
    Cuda;
    Vulkan;
    WebGPU;
}
