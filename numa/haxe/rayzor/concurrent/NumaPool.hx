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
 * Behaviour:
 * - Single-node (macOS, single-socket Linux, wasm, Windows with one
 *   node): no fanout — runs inline on the calling thread.
 * - Multi-node hardware: also inline today. Multi-node fanout via
 *   Thread.spawn is a tracked roadmap item; the API surface below is
 *   stable so call sites won't change when fanout lands.
 *
 * Example:
 * ```haxe
 * var pool = NumaPool.global();
 * pool.parallelFor(1000000, function(idx:Int, node:Int):Void {
 *     // ...
 * });
 * ```
 */
class NumaPool {
    private var _nodeCount:Int;

    public function new() {
        _nodeCount = NumaTopology.nodeCount();
    }

    /** Number of NUMA nodes this pool fans out across. */
    public function nodeCount():Int return _nodeCount;

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
     * attention heads) — lets the inner loop be tightly SIMD-optimised.
     */
    public function parallelRows(rows:Int, fn:Int->Int->Int->Void):Void {
        if (rows <= 0) return;
        fn(0, rows, 0);
    }

    private static var _instance:NumaPool;
    /** Process-wide singleton pool, lazily constructed on first call. */
    public static function global():NumaPool {
        if (_instance == null) _instance = new NumaPool();
        return _instance;
    }
}
