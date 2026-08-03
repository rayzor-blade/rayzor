package rayzor.gpu;

/**
 * Opaque handle to a GPU-resident buffer.
 *
 * Created via `GPUCompute.createBuffer()` or `GPUCompute.allocBuffer()`.
 * Data can be read back to a CPU tensor via `GPUCompute.toTensor()`.
 *
 * GpuBuffer is an opaque pointer — all operations go through the
 * GPUCompute context that created it.
 */
@:native("rayzor::gpu::GpuBuffer")
extern class GpuBuffer {
    /** Get the number of elements in this buffer. */
    @:native("rayzor_gpu_compute_buffer_numel")
    public function numel():Int;

    /** Get the dtype tag of this buffer. */
    @:native("rayzor_gpu_compute_buffer_dtype")
    public function dtype():rayzor.ds.DType;

    /**
     * Element-wise GPU addition (`a + b`).
     * Returns a new lazy GpuBuffer; the underlying kernel runs when the
     * result is read back via `GPUCompute.toTensor` / `.sum` / etc.
     */
    @:native("rayzor_gpu_buffer_add")
    @:op(A + B)
    public function add(other:GpuBuffer):GpuBuffer;

    /** Element-wise GPU subtraction (`a - b`). */
    @:native("rayzor_gpu_buffer_sub")
    @:op(A - B)
    public function sub(other:GpuBuffer):GpuBuffer;

    /** Element-wise GPU multiplication (`a * b`). */
    @:native("rayzor_gpu_buffer_mul")
    @:op(A * B)
    public function mul(other:GpuBuffer):GpuBuffer;

    /** Element-wise GPU division (`a / b`). */
    @:native("rayzor_gpu_buffer_div")
    @:op(A / B)
    public function div(other:GpuBuffer):GpuBuffer;
}
