package rayzor.concurrent;

/**
 * NUMA-aware thread pool for data-parallel work.
 *
 * `NumaPool` is the canonical entry point for CPU-parallel kernels in
 * rayzor / nue — anything that iterates over a large index range (matmul
 * rows, conv tiles, attention heads, elementwise sweeps, KV-cache fills,
 * etc.) should route through `NumaPool.global().parallelFor(...)` so
 * workers stay pinned to their NUMA node and produce node-local
 * first-touch allocations.
 *
 * Degenerate paths:
 * - `nodeCount() == 1` (macOS, single-socket Linux, wasm, Windows with
 *   one node): inline execution on the calling thread. Same Haxe source
 *   compiles unchanged across platforms.
 * - On every platform today: also inline. Multi-node fanout is a
 *   tracked roadmap item; the API surface below is stable so call
 *   sites won't change when fanout lands.
 *
 * Example:
 * ```haxe
 * import rayzor.concurrent.NumaPool;
 *
 * var pool = NumaPool.global();
 * trace("running on " + pool.nodeCount() + " nodes");
 *
 * pool.parallelFor(1000000, function(idx:Int, node:Int):Void {
 *     // work on a NUMA-pinned worker (when fanout lands; today inline).
 * });
 * ```
 */
class NumaPool {
    private var _nodeCount:Int;

    public function new() {
        _nodeCount = NumaTopology.nodeCount();
    }

    /** Number of NUMA nodes this pool fans out across. */
    public function nodeCount():Int {
        return _nodeCount;
    }

    /**
     * Iterate `items` indices with the user closure `fn(idx, node)`,
     * partitioned across NUMA nodes.
     */
    public function parallelFor(items:Int, fn:Int->Int->Void):Void {
        if (items <= 0) return;
        var i = 0;
        while (i < items) {
            fn(i, 0);
            i++;
        }
    }

    /**
     * Block-partitioned variant: invoke `fn(rowStart, rowEnd, node)` once
     * per node with a contiguous half-open block range.
     *
     * Use when the inner work is itself a loop (matmul rows, conv tiles,
     * attention heads) and you want one call per chunk instead of one
     * call per row — lets the inner loop be tightly SIMD-optimised.
     */
    public function parallelRows(rows:Int, fn:Int->Int->Int->Void):Void {
        if (rows <= 0) return;
        fn(0, rows, 0);
    }

    private static var _instance:NumaPool;
    /** Process-wide singleton pool, lazily constructed on first call. */
    public static function global():NumaPool {
        if (_instance == null) {
            _instance = new NumaPool();
        }
        return _instance;
    }
}
