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
@:derive([Clone])
@:shared
@:native("rayzor::ds::QTensor")
extern class QTensor {
    /**
     * Quantise an f32 source tensor (must be 2-D, shape `[rows, cols]`)
     * to the requested scheme. Returns a fresh `QTensor` owning the
     * quantised bytes.
     *
     * For `Q4_K_M`, `cols` must be a multiple of 256.
     */
    @:native("rayzor_qtensor_from_f32_int8_t")
    public static function fromFloat32(src:Tensor, scheme:QScheme):QTensor;

    /**
     * Wrap a pre-quantised Q4_K_M byte buffer. The intended caller is a
     * GGUF loader handing the runtime a raw block pointer. When
     * `takeOwnership` is true, the runtime will `free()` the buffer
     * with the `QTensor`; pass false for mmap-backed buffers.
     */
    @:native("rayzor_qtensor_wrap_q4_k_m")
    public static function wrapQ4KM(blockData:Ptr<Float>, rows:Int, cols:Int, takeOwnership:Int):QTensor;

    /**
     * Build a Q4_K_M QTensor from a `haxe.io.Bytes` whose underlying buffer
     * contains a contiguous sequence of `(rows * cols / 256)` Q4_K_M
     * super-blocks (144 bytes each). The runtime wraps the source bytes
     * zero-copy, so the source buffer / mmap must stay alive as long as the
     * QTensor.
     *
     * Intended caller: the GGUF loader handing the runtime a tensor slice
     * cut out of the on-disk weights file.
     */
    @:native("rayzor_qtensor_from_bytes_q4_k_m")
    public static function fromBytesQ4KM(bytes:haxe.io.Bytes, rows:Int, cols:Int):QTensor;

    /**
     * Build a Q6_K QTensor. Same shape/semantics as `fromBytesQ4KM` but for
     * GGUF dtype 14, which uses 210-byte super-blocks. Used by `Q4_K_M`
     * GGUF variants for the token-embedding, attention-V, and FFN-down
     * weights where Q6_K accuracy is preferred over Q4_K compression.
     */
    @:native("rayzor_qtensor_from_bytes_q6_k")
    public static function fromBytesQ6K(bytes:haxe.io.Bytes, rows:Int, cols:Int):QTensor;

    /**
     * Re-quantise a Q6_K tensor as Q4_K_M. Returns a fresh owning QTensor;
     * the source is unchanged. Use case: moving the lm_head off the Q6_K
     * SDOT path (which still pays the 6-bit reconstruction overhead per
     * block) onto the faster Q4_K_M SDOT path, at a small per-element
     * quantisation loss.
     *
     * Returns null on gate violation: source must be Q6_K, rows × cols
     * must be a multiple of 256, and allocation must succeed. Caller
     * should fall back to the original tensor in that case.
     */
    @:native("rayzor_qtensor_requant_q6k_to_q4km")
    public function requantQ6KToQ4KM():QTensor;

    /** Cheap @:shared handle clone (Arc refcount bump). No @:native — the
        call routes through the @:derive(Clone)/@:shared intercept; this
        declaration exists so the RETURN TYPE is known at any compile order. */
    @:native("rayzor_qtensor_clone")
    public function clone():QTensor;

    /** Disjoint-storage deep copy. */
    @:native("rayzor_qtensor_deep_clone")
    public function deepClone():QTensor;

    /** Number of rows in this 2-D matrix. */
    @:native("rayzor_qtensor_rows")
    public function rows():Int;

    /** Number of columns in this 2-D matrix. */
    @:native("rayzor_qtensor_cols")
    public function cols():Int;

    /**
        Raw base address of the quantised weight bytes, as a `Usize`. Lets a
        pure-Haxe qmatmul read Q4_K_M super-blocks directly out of the weight
        buffer (the per-block dot leaves Rust FFI):

        ```haxe
        var base = qt.dataPtr();
        var bpr = qt.cols() >> 8; // blocks-per-row = cols / 256
        // row r, block b begins at base + (r*bpr + b)*144 bytes
        var blk = SIMD16i8.load(Ptr.fromRaw(base + Usize.fromInt(((r*bpr + b)*144) + 16)));
        ```

        Guest-resident on wasm (the weight buffer lives in guest linear memory),
        so the offset is a valid guest `SIMD16i8.load` address.
    **/
    @:native("rayzor_qtensor_data_ptr")
    public function dataPtr():Usize;

    /** Total element count (rows * cols). */
    @:native("rayzor_qtensor_numel")
    public function numel():Int;

    /** Quantisation scheme. */
    @:native("rayzor_qtensor_scheme")
    public function scheme():QScheme;

    /**
     * Dequant the whole tensor into a fresh f32 `Tensor`. Used for
     * accuracy comparison or debug — production code should prefer
     * `matmulF32` which fuses dequant + matmul.
     */
    @:native("rayzor_qtensor_dequant")
    public function dequant():Tensor;

    /**
     * Fused dequant-matmul: `self [M, K] × b [K, N] → out [M, N]` where
     * `self` is quantised and `b` is f32. Returns a fresh f32 `Tensor`.
     */
    @:native("rayzor_qtensor_matmul_f32")
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
    @:native("rayzor_tensor_matmul_qt_t_f32")
    public function matmulXTQ(x:Tensor):Tensor;

    /**
     * Threaded chunk variant: fills rows `[nStart, nEnd)` of a pre-
     * allocated F32 result tensor `y[B, N]` with the same dot products
     * `matmulXTQ` computes. Workers split disjoint `[nStart, nEnd)`
     * ranges and write to non-overlapping columns of `y`, so no
     * synchronisation is needed beyond the caller's fork-join barrier.
     *
     * Used by `rayzor.concurrent.WorkerPool.parallelRows` to multi-thread
     * the large projection matmuls (Q/K/V/O, FFN gate/up/down) without
     * a hidden Rust thread pool.
     *
     * Returns 1 on success, 0 on shape mismatch / null input.
     */
    @:native("rayzor_tensor_matmul_qt_t_f32_chunk")
    public function matmulXTQChunk(x:Tensor, y:Tensor, nStart:Int, nEnd:Int):Int;

    /**
     * Threaded `y = x @ self.T`: per-call fork-join over output rows
     * via `std::thread::scope` in the runtime. `threads = 0` picks a
     * default (currently 6, sized for M1 Pro's 8 P-cores with
     * headroom). Workers split disjoint output-row ranges; the join
     * is implicit at the scope boundary, no pool outlives the call.
     *
     * This is the established default threading path. The pure-Haxe
     * `WorkerPool.parallelRows` band-loop route (`matmulXTQChunk`) is the
     * alternative; an earlier JIT trap-stub cascade when importing the pool
     * from `nue.Linear` was the reason it was avoided — re-verify it is
     * resolved (the WorkerPool fix chain likely closed it) before switching
     * Linear's forward over.
     *
     * Returns a fresh F32 `Tensor`; null on shape mismatch.
     */
    @:native("rayzor_tensor_matmul_qt_t_f32_threaded")
    public function matmulXTQThreaded(x:Tensor, threads:Int):Tensor;

    /**
     * Fused Q/K/V projection: computes `y_q = x @ qW.T`, `y_k = x @ kW.T`,
     * and `y_v = x @ vW.T` against three Q4_K_M weights in a single
     * dispatch. The activation X is pre-quantised to Q8_K exactly once
     * and the three projections share that view; a single
     * `parallel_rows` fan-out covers the concatenated
     * `[0, q_n + k_n + v_n)` row space, replacing three sequential
     * fork-joins with one. Output rows are disjoint across workers and
     * disjoint across the three result tensors.
     *
     * Returns a fresh 3-element `Array<Tensor>` of `[Q, K, V]` handles
     * on success. On gate-miss (SDOT unavailable, batch != 1, not all
     * three weights Q4_K_M, X non-contiguous) the array carries
     * `[null, null, null]`; check `arr[0] == null` and fall back to
     * three sequential `matmulXTQThreaded` calls.
     *
     * `threads = 0` picks the auto default (currently 6, sized for
     * M1 Pro). Reduction order is byte-identical to three separate
     * `matmulXTQThreaded` calls.
     */
    /** Writes the three result handles into the caller-provided `out`
     *  (pre-sized `[null, null, null]`). Slots stay null on gate-miss —
     *  null-check `out[0]` and fall back to three `matmulXTQThreaded`
     *  calls. Returns 0 on success. */
    @:native("rayzor_tensor_matmul_qkv_fused_arr")
    public static function fusedQkvIntoArr(x:Tensor, qW:QTensor, kW:QTensor, vW:QTensor, threads:Int, out:Array<Tensor>):Int;

    /**
     * Q6_K-aware row gather: dequant just the rows named by `indices`
     * out of this `[N, cols]` Q6_K matrix into a fresh f32
     * `[indices.length, cols]` `Tensor`. The runtime requires this
     * tensor's scheme to be Q6_K and `cols` to be a whole number of
     * 256-element super-blocks; out-of-range indices leave the
     * corresponding output row zero-filled.
     *
     * Used by `nue.Embedding` to turn token IDs into per-token
     * embeddings against a Q6_K token-embedding weight, avoiding a
     * full `[vocab, hidden]` dequant per forward pass.
     */
    @:native("rayzor_tensor_gather_rows_q6_k_arr")
    public function gatherRowsQ6K(indices:Array<Int>):Tensor;

    /** Free the quantised storage. */
    @:native("rayzor_qtensor_free")
    public function free():Void;
}
