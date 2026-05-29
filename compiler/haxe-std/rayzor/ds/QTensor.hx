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

    /**
     * Build a Q6_K QTensor. Same shape/semantics as `fromBytesQ4KM` but for
     * GGUF dtype 14, which uses 210-byte super-blocks. Used by `Q4_K_M`
     * GGUF variants for the token-embedding, attention-V, and FFN-down
     * weights where Q6_K accuracy is preferred over Q4_K compression.
     */
    @:native("qtensor_from_bytes_q6_k")
    public static function fromBytesQ6K(bytes:haxe.io.Bytes, rows:Int, cols:Int):QTensor;

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

    /**
     * Linear-style fused matmul: `out[B, N] = x[B, K] × self[N, K].T`, with
     * `self` Q4_K_M `[N, K]` (rows=N=out, cols=K=in; blocks along K). This
     * is the operation a PyTorch `nn.Linear` performs against its weight,
     * skipping the F32 dequant entirely. Dequant happens one Wq row at a
     * time into a small scratch buffer (amortised across the batch).
     *
     * Returns a fresh f32 `Tensor`; null if shapes mismatch.
     */
    @:native("tensor_matmul_qt_t_f32")
    public function matmulXTQ(x:Tensor):Tensor;

    /**
     * Threaded chunk variant: fills rows `[nStart, nEnd)` of a pre-
     * allocated F32 result tensor `y[B, N]` with the same dot products
     * `matmulXTQ` computes. Workers split disjoint `[nStart, nEnd)`
     * ranges and write to non-overlapping columns of `y`, so no
     * synchronisation is needed beyond the caller's fork-join barrier.
     *
     * Used by `rayzor.concurrent.NumaPool.parallelRows` to multi-thread
     * the large projection matmuls (Q/K/V/O, FFN gate/up/down) without
     * a hidden Rust thread pool.
     *
     * Returns 1 on success, 0 on shape mismatch / null input.
     */
    @:native("tensor_matmul_qt_t_f32_chunk")
    public function matmulXTQChunk(x:Tensor, y:Tensor, nStart:Int, nEnd:Int):Int;

    /**
     * Threaded `y = x @ self.T`: per-call fork-join over output rows
     * via `std::thread::scope` in the runtime. `threads = 0` picks a
     * default (currently 6, sized for M1 Pro's 8 P-cores with
     * headroom). Workers split disjoint output-row ranges; the join
     * is implicit at the scope boundary, no pool outlives the call.
     *
     * This is the practical threading path while the Haxe-side
     * `NumaPool` import currently triggers a JIT trap-stub cascade
     * when imported from `nue.Linear`.
     *
     * Returns a fresh F32 `Tensor`; null on shape mismatch.
     */
    @:native("tensor_matmul_qt_t_f32_threaded")
    public function matmulXTQThreaded(x:Tensor, threads:Int):Tensor;

    /** Free the quantised storage. */
    @:native("qtensor_free")
    public function free():Void;
}
