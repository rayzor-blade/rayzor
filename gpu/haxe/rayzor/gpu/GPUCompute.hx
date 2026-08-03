package rayzor.gpu;

/**
 * GPU compute context for accelerated numerical operations.
 *
 * GPUCompute provides GPU-accelerated buffer management and (in future phases)
 * elementwise operations, reductions, and linear algebra via Metal/CUDA/WebGPU.
 *
 * This is an opt-in native package — it requires the rayzor-gpu dynamic library
 * to be available at runtime. Use `GPUCompute.isAvailable()` to check.
 *
 * Example:
 * ```haxe
 * import rayzor.gpu.GPUCompute;
 * import rayzor.ds.Tensor;
 *
 * if (GPUCompute.isAvailable()) {
 *     var gpu = GPUCompute.create();
 *     var t = Tensor.ones([1024], F32);
 *     var buf = gpu.createBuffer(t);
 *     var t2 = gpu.toTensor(buf);
 *     trace(t2.sum());  // 1024.0
 *     gpu.freeBuffer(buf);
 *     gpu.destroy();
 * }
 * ```
 */
@:native("rayzor::gpu::GPUCompute")
extern class GPUCompute {
    /** Create a new GPU compute context. Returns null if GPU is unavailable. */
    @:native("rayzor_gpu_compute_create")
    public static function create():GPUCompute;

    /** Destroy this GPU compute context and release device resources. */
    @:native("rayzor_gpu_compute_destroy")
    public function destroy():Void;

    /** Check if GPU compute is available on this system. */
    @:native("rayzor_gpu_compute_is_available")
    public static function isAvailable():Bool;

    /** Create a GPU buffer by copying data from a CPU tensor. */
    @:native("rayzor_gpu_compute_create_buffer")
    public function createBuffer(tensor:rayzor.ds.Tensor):GpuBuffer;

    /** Allocate an empty GPU buffer with the given element count and dtype. */
    @:native("rayzor_gpu_compute_alloc_buffer")
    public function allocBuffer(numel:Int, dtype:rayzor.ds.DType):GpuBuffer;

    /** Copy GPU buffer data back to a new CPU tensor. */
    @:native("rayzor_gpu_compute_to_tensor")
    public function toTensor(buffer:GpuBuffer):rayzor.ds.Tensor;

    /** Free a GPU buffer. */
    @:native("rayzor_gpu_compute_free_buffer")
    public function freeBuffer(buffer:GpuBuffer):Void;

    // -- Binary elementwise ops: result[i] = a[i] OP b[i] -------------------

    /** GPU-accelerated elementwise addition. */
    @:native("rayzor_gpu_compute_add")
    public function add(a:GpuBuffer, b:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise subtraction. */
    @:native("rayzor_gpu_compute_sub")
    public function sub(a:GpuBuffer, b:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise multiplication. */
    @:native("rayzor_gpu_compute_mul")
    public function mul(a:GpuBuffer, b:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise division. */
    @:native("rayzor_gpu_compute_div")
    public function div(a:GpuBuffer, b:GpuBuffer):GpuBuffer;

    // -- Unary elementwise ops: result[i] = OP(a[i]) ------------------------

    /** GPU-accelerated elementwise negation. */
    @:native("rayzor_gpu_compute_neg")
    public function neg(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise absolute value. */
    @:native("rayzor_gpu_compute_abs")
    public function abs(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise square root. */
    @:native("rayzor_gpu_compute_sqrt")
    public function sqrt(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise exponential (e^x). */
    @:native("rayzor_gpu_compute_exp")
    public function exp(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise natural logarithm. */
    @:native("rayzor_gpu_compute_log")
    public function log(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise ReLU: max(0, x). */
    @:native("rayzor_gpu_compute_relu")
    public function relu(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise sigmoid: 1 / (1 + exp(-x)). */
    @:native("rayzor_gpu_compute_sigmoid")
    public function sigmoid(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise tanh. */
    @:native("rayzor_gpu_compute_tanh")
    public function tanh(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise GELU activation. */
    @:native("rayzor_gpu_compute_gelu")
    public function gelu(a:GpuBuffer):GpuBuffer;

    /** GPU-accelerated elementwise SiLU (Swish): x * sigmoid(x). */
    @:native("rayzor_gpu_compute_silu")
    public function silu(a:GpuBuffer):GpuBuffer;

    // -- Reductions: buffer -> scalar ----------------------------------------

    /** Sum all elements in a GPU buffer. */
    @:native("rayzor_gpu_compute_sum")
    public function sum(buf:GpuBuffer):Float;

    /** Mean of all elements in a GPU buffer. */
    @:native("rayzor_gpu_compute_mean")
    public function mean(buf:GpuBuffer):Float;

    /** Maximum element in a GPU buffer. */
    @:native("rayzor_gpu_compute_max")
    public function max(buf:GpuBuffer):Float;

    /** Minimum element in a GPU buffer. */
    @:native("rayzor_gpu_compute_min")
    public function min(buf:GpuBuffer):Float;

    // -- Linear algebra ------------------------------------------------------

    /** Dot product of two GPU buffers (elementwise multiply + sum). */
    @:native("rayzor_gpu_compute_dot")
    public function dot(a:GpuBuffer, b:GpuBuffer):Float;

    /** Matrix multiplication: C(M×N) = A(M×K) × B(K×N). */
    @:native("rayzor_gpu_compute_matmul")
    public function matmul(a:GpuBuffer, b:GpuBuffer, m:Int, k:Int, n:Int):GpuBuffer;

    /** Batched matrix multiplication: C[b](M×N) = A[b](M×K) × B[b](K×N) for b in 0..batch. */
    @:native("rayzor_gpu_compute_batch_matmul")
    public function batchMatmul(a:GpuBuffer, b:GpuBuffer, batch:Int, m:Int, k:Int, n:Int):GpuBuffer;

    // -- Transformer primitives ----------------------------------------------

    /**
     * RMS normalization with a learnable per-channel gain.
     * `y[..., i] = x[..., i] / sqrt(mean(x²) + eps) * weight[i]`.
     * `rowLen` is the trailing-dim length (hidden_size); the input is
     * treated as `[groups, rowLen]` with `groups = x.numel / rowLen`.
     */
    @:native("rayzor_gpu_compute_rms_norm")
    public function rmsNorm(x:GpuBuffer, weight:GpuBuffer, rowLen:Int, eps:Float):GpuBuffer;

    /**
     * Apply rotary position embedding to `x [seqLen, numHeads, headDim]`
     * using precomputed cos/sin LUTs of shape `[cosMaxSeq, headDim/2]`.
     * `positionOffset` shifts the per-row absolute position (0 for
     * prefill, positive for decode).
     */
    @:native("rayzor_gpu_compute_rope")
    public function rope(
        x:GpuBuffer, cos:GpuBuffer, sin:GpuBuffer,
        seqLen:Int, numHeads:Int, headDim:Int,
        positionOffset:Int, cosMaxSeq:Int
    ):GpuBuffer;

    // -- Structured buffer ops (@:gpuStruct) -----------------------------------

    /** Create a GPU buffer from an array of @:gpuStruct instances. */
    @:native("rayzor_gpu_compute_create_struct_buffer")
    public function createStructBuffer(array:Dynamic, count:Int, structSize:Int):GpuBuffer;

    /** Allocate an empty GPU buffer for `count` structs of `structSize` bytes. */
    @:native("rayzor_gpu_compute_alloc_struct_buffer")
    public function allocStructBuffer(count:Int, structSize:Int):GpuBuffer;

    /** Read a float field from a structured buffer (returns Float). */
    @:native("rayzor_gpu_compute_read_struct_float")
    public function readStructFloat(buffer:GpuBuffer, index:Int, structSize:Int, fieldOffset:Int):Float;

    /** Read an int field from a structured buffer (returns Int). */
    @:native("rayzor_gpu_compute_read_struct_int")
    public function readStructInt(buffer:GpuBuffer, index:Int, structSize:Int, fieldOffset:Int):Int;
}
