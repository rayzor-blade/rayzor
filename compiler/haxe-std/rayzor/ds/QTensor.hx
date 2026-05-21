package rayzor.ds;

/**
 * Quantised 2-D tensor for memory-efficient LLM inference.
 *
 * Storage is compressed to INT8 or Q4_K_M; arithmetic happens via fused
 * dequant-on-the-fly kernels that never materialise an f32 copy of the
 * weights. The typical use is the model's weight matrices (Linear, GQA
 * projections, FFN up/down) — kept in QTensor form, multiplied against
 * f32 activations.
 *
 * Example:
 * ```haxe
 * // Quantise an f32 weight matrix to INT8.
 * var w_f32 = Tensor.rand([4096, 4096], F32);
 * var w_q = QTensor.fromFloat32(w_f32, INT8);
 *
 * // Forward pass: x: [batch, 4096] @ w: [4096, 4096] → [batch, 4096]
 * var y = w_q.matmulF32(x);
 *
 * // Inspect storage savings.
 * trace("rows=" + w_q.rows() + " cols=" + w_q.cols());
 * trace("scheme=" + w_q.scheme());
 * ```
 *
 * The runtime representation is an opaque pointer; lifetime is managed
 * via explicit `free()`. (Refcounted lifetime is a future refinement.)
 */
@:native("rayzor::ds::QTensor")
extern class QTensor {
    /**
     * Quantise an f32 source tensor (must be 2-D, shape `[rows, cols]`)
     * to the requested scheme. Returns a fresh `QTensor` owning the
     * quantised bytes.
     *
     * For `Q4_K_M`, `cols` must be a multiple of 256.
     */
    @:native("qtensor_from_float32")
    public static function fromFloat32(src:Tensor, scheme:QScheme):QTensor;

    /**
     * Wrap a pre-quantised Q4_K_M byte buffer. The intended caller is a
     * GGUF loader handing the runtime a raw block pointer. When
     * `takeOwnership` is true, the runtime will `free()` the buffer
     * with the `QTensor`; pass false for mmap-backed buffers.
     */
    @:native("qtensor_wrap_q4_k_m")
    public static function wrapQ4KM(blockData:Ptr<Float>, rows:Int, cols:Int, takeOwnership:Int):QTensor;

    /**
     * Build a Q4_K_M QTensor from a `haxe.io.Bytes` whose underlying buffer
     * contains a contiguous sequence of `(rows * cols / 256)` Q4_K_M
     * super-blocks (144 bytes each). The runtime copies the bytes into an
     * owning buffer, so the source `Bytes` can be freed after this returns.
     *
     * Intended caller: the GGUF loader handing the runtime a tensor slice
     * cut out of the on-disk weights file.
     */
    @:native("qtensor_from_bytes_q4_k_m")
    public static function fromBytesQ4KM(bytes:haxe.io.Bytes, rows:Int, cols:Int):QTensor;

    /** Number of rows in this 2-D matrix. */
    @:native("qtensor_rows")
    public function rows():Int;

    /** Number of columns in this 2-D matrix. */
    @:native("qtensor_cols")
    public function cols():Int;

    /** Total element count (rows * cols). */
    @:native("qtensor_numel")
    public function numel():Int;

    /** Quantisation scheme. */
    @:native("qtensor_scheme")
    public function scheme():QScheme;

    /**
     * Dequant the whole tensor into a fresh f32 `Tensor`. Used for
     * accuracy comparison or debug — production code should prefer
     * `matmulF32` which fuses dequant + matmul.
     */
    @:native("qtensor_dequant")
    public function dequant():Tensor;

    /**
     * Fused dequant-matmul: `self [M, K] × b [K, N] → out [M, N]` where
     * `self` is quantised and `b` is f32. Returns a fresh f32 `Tensor`.
     */
    @:native("qtensor_matmul_f32")
    public function matmulF32(b:Tensor):Tensor;

    /** Free the quantised storage. */
    @:native("qtensor_free")
    public function free():Void;
}
