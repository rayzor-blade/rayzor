package rayzor.ds;

/**
 * N-dimensional tensor with shape, strides, and dtype.
 *
 * Tensor is the fundamental data structure for numerical computing in Rayzor.
 * It supports arbitrary shapes, strided views (reshape/transpose without copy),
 * and element-wise + linear algebra operations.
 *
 * The runtime representation is a heap-allocated struct with reference counting
 * for shared views. All operations return new tensors (immutable value semantics).
 *
 * Example:
 * ```haxe
 * var a = Tensor.zeros([2, 3], F32);
 * var b = Tensor.ones([2, 3], F32);
 * var c = a + b;
 * trace(c.sum());
 * ```
 */
@:derive([Clone])
@:shared
@:native("rayzor::ds::Tensor")
extern class Tensor {
    // --- Construction ---

    /** Create a tensor filled with zeros */
    @:native("tensor_zeros")
    public static function zeros(shape:Array<Int>, dtype:DType):Tensor;

    /** Create a tensor filled with ones */
    @:native("tensor_ones")
    public static function ones(shape:Array<Int>, dtype:DType):Tensor;

    /** Create a tensor filled with a constant value */
    @:native("tensor_full")
    public static function full(shape:Array<Int>, value:Float, dtype:DType):Tensor;

    /** Create a 1-D tensor from a flat array of floats */
    @:native("tensor_fromArray")
    public static function fromArray(data:Array<Float>, dtype:DType):Tensor;

    /**
     * Build an F32 Tensor of shape `shape` by widening a contiguous
     * little-endian IEEE 754 half-precision (F16) byte buffer to f32.
     * Output is always F32 (Phase-3 compute kernels for F16 aren't
     * shipped yet; widening at load keeps the rest of the stack
     * dtype-uniform).
     */
    @:native("tensor_fromBytesF16")
    public static function fromBytesF16(bytes:haxe.io.Bytes, shape:Array<Int>):Tensor;

    /**
     * Build an F32 Tensor by memcpying raw F32 bytes (little-endian) with
     * the given shape. Used by the GGUF loader for dtype=0 weights.
     * Bypasses the `fromArray(Array<Float>, DType.F32)` round-trip whose
     * `Array<Float>.push` wrapper loses precision when crossing the i64
     * extern boundary.
     */
    @:native("tensor_fromBytesF32")
    public static function fromBytesF32(bytes:haxe.io.Bytes, shape:Array<Int>):Tensor;

    /**
     * Build an F32 Tensor of shape `shape` by dequantising a GGML Q8_0
     * byte buffer (32-element blocks of `[f16 scale][32 × i8]`, 34 bytes
     * each). Output is F32 — Q8_0 is rare in Q4_K_M-quantised models so
     * load-time expansion is the simplest path.
     */
    @:native("tensor_fromBytesQ8_0")
    public static function fromBytesQ8_0(bytes:haxe.io.Bytes, shape:Array<Int>):Tensor;

    /** Create a tensor with random values in [0, 1) */
    @:native("tensor_rand")
    public static function rand(shape:Array<Int>, dtype:DType):Tensor;

    // --- Properties ---

    /** Get the shape as an array of dimension sizes */
    @:native("tensor_shape")
    public function shape():Array<Int>;

    /** Number of dimensions */
    @:native("tensor_ndim")
    public function ndim():Int;

    /** Total number of elements */
    @:native("tensor_numel")
    public function numel():Int;

    /** Element data type */
    @:native("tensor_dtype")
    public function dtype():DType;

    /**
     * Device tag this tensor lives on, as a raw int:
     * `0=CPU, 1=Metal, 2=Cuda, 3=Vulkan, 4=WebGPU`. For a typed `Device`
     * value, call `Tensors.deviceOf(t)` (sibling helper, future). Phase 1a:
     * every constructor on this class returns a tensor with `deviceTag() == 0`.
     */
    @:native("tensor_device")
    public function deviceTag():Int;

    /**
     * NUMA node hint (meaningful only when device tag is CPU). `-1` means
     * "any node" (no affinity); `>= 0` is a specific node from
     * `rayzor.concurrent.NumaTopology`.
     */
    @:native("tensor_numa_node")
    public function numaNode():Int;

    // --- Element access ---

    /** Get element at indices */
    @:native("tensor_get")
    public function get(indices:Array<Int>):Float;

    /**
     * Flat-indexed scalar read. Skips the `Array<Int>` allocation in
     * `get()` — for a 128k-element logits vector this is ~100× cheaper
     * per call. Use when scanning a contiguous tensor linearly (e.g.
     * top-k selection over a logits row).
     */
    @:native("tensor_get_flat")
    public function getFlat(i:Int):Float;

    /**
     * Single-FFI top-K + repetition-penalty scan over the tensor.
     *
     * Walks every element of `this`, applies a repetition penalty when
     * the element index appears in `recentIds`, and insertion-sorts the
     * top `k` (logit, id) pairs into the pre-allocated `outLogits` /
     * `outIds` buffers (descending by logit). Returns the number of
     * survivors actually written (≤ `k`).
     *
     * Replaces the per-element `tensor.getFlat(i)` loop in samplers
     * that scan a 128k-vocab logits vector — eliminates 128k FFI
     * boundary crossings per token in exchange for one bulk-scan call
     * whose inner loop is a tight Rust function.
     *
     * Constraints (`-1` returned on violation; caller should fall back
     * to the per-element scan):
     *   - tensor dtype must be `F32`
     *   - tensor data must be contiguous (owns_data)
     *   - `outLogits.length >= k`, `outIds.length >= k`
     *
     * `recentIds` may be `[]` and `penalty <= 1.0` to disable the
     * repetition-penalty step entirely — the scan still runs and the
     * top-K is returned untouched.
     */
    @:native("tensor_topk_scan")
    public function topkScan(
        outLogits:Array<Float>,
        outIds:Array<Int>,
        k:Int,
        recentIds:Array<Int>,
        penalty:Float
    ):Int;

    /** Set element at indices */
    @:native("tensor_set")
    public function set(indices:Array<Int>, value:Float):Void;

    /**
     * Append the rows of `src` into this tensor starting at row `dstRowOffset`
     * along axis 0. Returns `0` on success, `-1` on failure (shape/dtype
     * mismatch, out-of-bounds, etc.). Callers must treat `-1` as a hard
     * failure — there is no scalar fallback.
     */
    @:native("tensor_append_along_0_f32")
    public function appendAlong0(src:Tensor, dstRowOffset:Int):Int;

    /**
     * Broadcast-repeat `src` along axis 0 into this tensor `repeats` times.
     * Returns `0` on success, `-1` on failure. Callers must treat `-1` as a
     * hard failure — there is no scalar fallback.
     */
    @:native("tensor_broadcast_repeat_0_f32")
    public function broadcastRepeat0(src:Tensor, repeats:Int):Int;

    // --- Reshape / view (no copy) ---

    /** Reshape to a new shape (same numel) */
    @:native("tensor_reshape")
    public function reshape(shape:Array<Int>):Tensor;

    /** 2D matrix transpose */
    @:native("tensor_transpose")
    public function transpose():Tensor;

    /** N-D permutation (no copy, view) */
    @:native("tensor_permute")
    public function permute(axes:Array<Int>):Tensor;

    /** Slice along a single dim, [start, end) (no copy, view) */
    @:native("tensor_slice")
    public function slice(dim:Int, start:Int, end:Int):Tensor;

    /** Cheap @:shared handle clone (Arc refcount bump). No @:native — the
        call routes through the @:derive(Clone)/@:shared intercept; this
        declaration exists so the RETURN TYPE is known at any compile order. */
    public function clone():Tensor;

    /** Disjoint-storage deep copy (cross-thread mutation patterns). */
    public function deepClone():Tensor;

    // --- Arithmetic (elementwise, return new tensor) ---

    /** Element-wise addition */
    @:native("tensor_add")
    @:op(A + B)
    public function add(other:Tensor):Tensor;

    /**
     * Adds src into this tensor in-place. Both tensors must be contiguous,
     * same shape, same dtype. Does NOT consume src (src remains live).
     * Use this in place of `var y = x.add(src);` to avoid the alloc+copy.
     */
    @:native("tensor_addInto")
    public function addInto(src:Tensor):Void;

    /** Element-wise subtraction */
    @:native("tensor_sub")
    @:op(A - B)
    public function sub(other:Tensor):Tensor;

    /** Element-wise multiplication */
    @:native("tensor_mul")
    @:op(A * B)
    public function mul(other:Tensor):Tensor;

    /** Element-wise division */
    @:native("tensor_div")
    @:op(A / B)
    public function div(other:Tensor):Tensor;

    // --- Linear algebra ---

    /** Matrix multiplication (2-D × 2-D → 2-D). */
    @:native("tensor_matmul")
    public function matmul(other:Tensor):Tensor;

    /**
     * Matmul with transposed RHS: `y[i, j] = sum_k a[i, k] * b[j, k]`.
     *
     * `self` is `[M, K]`, `other` is `[N, K]` (its second dim is the K of
     * matmul). Output is `[M, N]`. The natural shape for PyTorch-style
     * `Linear`: `y = x @ w.T` with `w[out, in]` and `x[batch, in]`.
     */
    @:native("tensor_matmul_t")
    public function matmulT(other:Tensor):Tensor;

    /**
     * Batched 3-D matrix multiplication. `self [batch, M, K]` ×
     * `other [batch, K, N]` → `[batch, M, N]`. Each batch slice runs an
     * independent matmul; SIMD axpy fast path on F32, scalar fallback on
     * other dtypes.
     */
    @:native("tensor_bmm")
    public function bmm(other:Tensor):Tensor;

    /**
     * Threaded batched matrix multiply. `threads = 0` picks auto fanout (6),
     * `threads = 1` falls through to sequential bmm. Output is byte-exact
     * with `bmm` (each row's dot product runs in a single thread; no
     * cross-thread reduction).
     */
    @:native("tensor_bmm_threaded")
    public function bmmThreaded(other:Tensor, threads:Int):Tensor;

    /**
     * Fused decode-step attention (single new query token, GQA-aware).
     *
     * `self` is the post-RoPE Q tensor of shape `[1, num_q_heads, head_dim]`,
     * F32, contiguous. `k` and `v` are the un-expanded KV cache **views** of
     * shape `[cache_len, num_kv_heads, head_dim]`, F32, with the inner two
     * axes contiguous (the standard `KVCache.slice(0, 0, currentLen)`
     * layout). `scale` is `1 / sqrt(head_dim)`.
     *
     * Returns a fresh `[1, num_q_heads, head_dim]` F32 owning tensor with
     * the attention output — same shape as `bmm(Q, K^T).softmax().bmm(V)`
     * but without materialising the expanded KV copies or the scores
     * tensor. Use this in place of the bmm+scale+softmax+bmm chain in
     * `GQAttention.forward` when `seqQ == 1` (decode step).
     *
     * Returns `null` if any gate fails (non-F32, non-contiguous Q,
     * seqQ != 1, K/V shapes inconsistent, GQA group doesn't divide).
     * Caller MUST handle null by falling back to the unfused path —
     * prefill (seqQ > 1) is intentionally not supported here.
     */
    @:native("tensor_flash_attn_decode")
    public function flashAttnDecode(k:Tensor, v:Tensor, scale:Float):Tensor;

    /**
     * GQA KV-head expansion. Self has shape `[seqK, num_kv_heads, head_dim]`
     * (KV-heads on axis 1, as produced by KVCache views). Returns a fresh
     * F32 tensor of shape `[num_kv_heads * repeats, seqK, head_dim]` with
     * `out[qh, j, d] = self[j, qh / repeats, d]`. Replaces the per-element
     * triple-loop expand pattern with a single strided memcpy per `(qh, j)`.
     * Returns null on dtype mismatch (F32 only), non-3D source, or non-
     * contiguous innermost dim.
     */
    @:native("tensor_expand_kv_heads_axis1")
    public function expandKvHeadsAxis1(repeats:Int):Tensor;

    /**
     * Fill the upper triangle of the last two dims with `-inf` so a
     * subsequent softmax row reads those positions as zero probability.
     * `positionOffset` shifts the diagonal — 0 for prefill, positive
     * for incremental decode where the new query is at logical position
     * `positionOffset`. Mutates in place; returns `this` for chaining.
     */
    @:native("tensor_causal_mask_")
    public function causalMask_(positionOffset:Int):Tensor;

    /**
     * Multiply every element by a scalar. Returns a new tensor; uses the
     * SIMD `mul_const_slice` fast path for F32 inputs.
     */
    @:native("tensor_scale")
    public function scale(factor:Float):Tensor;

    /**
     * Swap the last two dimensions (zero-copy view). Equivalent to
     * `permute([..., ndim-1, ndim-2])` but doesn't require an indices
     * array literal at the call site.
     */
    @:native("tensor_transpose_last2")
    public function transposeLast2():Tensor;

    /**
     * Row gather: pick the rows of this `[N, ...rest]` tensor named by
     * `indices` and stack them as `[indices.length, ...rest]`. Used by
     * `nue.Embedding` to turn token IDs into per-token embeddings.
     */
    @:native("tensor_gather_rows")
    public function gatherRows(indices:Array<Int>):Tensor;

    /** Dot product (flattened) */
    @:native("tensor_dot")
    public function dot(other:Tensor):Float;

    // --- Reductions ---

    /** Sum all elements (returns scalar tensor) */
    @:native("tensor_sum")
    public function sum():Float;

    /** Mean of all elements */
    @:native("tensor_mean")
    public function mean():Float;

    /** Maximum element */
    @:native("tensor_max")
    public function max():Float;

    /** Minimum element */
    @:native("tensor_min")
    public function min():Float;

    // --- Math ---

    /** Element-wise square root */
    @:native("tensor_sqrt")
    public function sqrt():Tensor;

    /** Element-wise exponential */
    @:native("tensor_exp")
    public function exp():Tensor;

    /** Element-wise natural logarithm */
    @:native("tensor_log")
    public function log():Tensor;

    /** Element-wise ReLU activation */
    @:native("tensor_relu")
    public function relu():Tensor;

    /** Element-wise GELU activation (tanh approximation) */
    @:native("tensor_gelu")
    public function gelu():Tensor;

    /** Element-wise SiLU / swish activation */
    @:native("tensor_silu")
    public function silu():Tensor;

    /** Softmax over the last dimension */
    @:native("tensor_softmax")
    public function softmax():Tensor;

    /** Layer normalization over the last dimension */
    @:native("tensor_layer_norm")
    public function layerNorm(eps:Float):Tensor;

    /** RMS normalization over the last dimension */
    @:native("tensor_rms_norm")
    public function rmsNorm(eps:Float):Tensor;

    /**
     * Apply rotary position embedding (RoPE) to a tensor of shape
     * `[seq_len, num_heads, head_dim]` (or 2-D `[seq_len, head_dim]`).
     *
     * `cos` and `sin` are precomputed lookup tables of shape
     * `[max_seq_len, head_dim/2]` (see `Tensor.ropeCosTable` /
     * `ropeSinTable`). `positionOffset` adds to the per-row position —
     * used by the KV-cache decode path to rotate the new query token at
     * its absolute position.
     */
    @:native("tensor_rope")
    public function rope(cos:Tensor, sin:Tensor, positionOffset:Int):Tensor;

    /**
     * Precomputed cosine table for RoPE. Shape `[maxSeqLen, headDim/2]`,
     * dtype F32. Pass to `Tensor.rope`. `base` is the frequency base
     * (10000.0 for standard Llama, 1000000.0 for long-context tunes).
     */
    @:native("tensor_rope_cos_table")
    public static function ropeCosTable(headDim:Int, maxSeqLen:Int, base:Float):Tensor;

    /**
     * Precomputed sine table for RoPE — companion to `ropeCosTable`.
     * Same shape and dtype.
     */
    @:native("tensor_rope_sin_table")
    public static function ropeSinTable(headDim:Int, maxSeqLen:Int, base:Float):Tensor;

    /**
     * F16-stored cosine LUT for RoPE — half the memory of the F32 variant.
     * Same `[maxSeqLen, headDim/2]` shape; precision loss bounded by f16
     * quantisation of `cos ∈ [-1, 1]` (≈5e-4 absolute), negligible for
     * inference. The GPU RoPE kernel reads through f32 anyway.
     */
    @:native("tensor_rope_cos_table_f16")
    public static function ropeCosTableF16(headDim:Int, maxSeqLen:Int, base:Float):Tensor;

    /** F16-stored sine LUT — companion to `ropeCosTableF16`. */
    @:native("tensor_rope_sin_table_f16")
    public static function ropeSinTableF16(headDim:Int, maxSeqLen:Int, base:Float):Tensor;

    // --- Interop ---

    /** Get raw data pointer for FFI */
    @:native("tensor_data")
    public function data():Ptr<Float>;

    /** Free tensor and its data */
    @:native("tensor_free")
    public function free():Void;
}
