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
 * - **Multi-node**: spawns one worker per NUMA node, each pinned via
 *   `NumaTopology.bindCurrent(node)` before invoking the user closure
 *   on its chunk. The user `fn` is captured by every worker — its
 *   captures must be `Send` (compiler validates at the spawn site).
 * - **Single-node** (macOS, single-socket Linux, wasm, Windows with
 *   one node): no fanout — runs inline on the calling thread.
 * - **Small work** (`items < nodeCount * 2`): also inline.
 *
 * Example:
 * ```haxe
 * var pool = NumaPool.global();
 * pool.parallelFor(1000000, function(idx:Int, node:Int):Void {
 *     // Each worker runs on a NUMA-pinned thread when nodeCount > 1.
 * });
 * ```
 */
class NumaPool {
    private var _nodeCount:Int;

    public function new() {
        _nodeCount = NumaTopology.nodeCount();
    }

    /**
     * Construct a pool with a forced node count, ignoring the runtime
     * topology. Useful for exercising the multi-node fanout path on
     * single-NUMA-node hardware and for unit testing. `bindCurrent`
     * becomes a soft-affinity hint on no-NUMA platforms, but the
     * `Thread.spawn` dispatch is real.
     */
    public static function withForcedNodes(nodes:Int):NumaPool {
        var p = new NumaPool();
        if (nodes > 0) p._nodeCount = nodes;
        return p;
    }

    /** Number of NUMA nodes this pool fans out across. */
    public function nodeCount():Int return _nodeCount;

    /**
     * Iterate `items` indices with the user closure `fn(idx, node)`,
     * partitioned across NUMA nodes.
     */
    public function parallelFor(items:Int, fn:Int->Int->Void):Void {
        if (items <= 0) return;

        if (_nodeCount <= 1 || items < _nodeCount * 2) {
            var i = 0;
            while (i < items) {
                fn(i, 0);
                i++;
            }
            return;
        }

        var nodes = _nodeCount;
        var chunkSize = Std.int(items / nodes);
        var threads = new Array<Thread<Int>>();

        for (n in 0...nodes) {
            var lo = n * chunkSize;
            var hi = (n == nodes - 1) ? items : (n + 1) * chunkSize;
            var node = n;
            var f = fn;
            var t = Thread.spawn(function():Int {
                NumaTopology.bindCurrent(node);
                var i = lo;
                while (i < hi) {
                    f(i, node);
                    i++;
                }
                return 0;
            });
            threads.push(t);
        }

        for (t in threads) {
            t.join();
        }
    }

    /**
     * Block-partitioned variant: invoke `fn(rowStart, rowEnd, node)` once
     * per node with a contiguous half-open block range.
     */
    public function parallelRows(rows:Int, fn:Int->Int->Int->Void):Void {
        if (rows <= 0) return;

        if (_nodeCount <= 1 || rows < _nodeCount) {
            fn(0, rows, 0);
            return;
        }

        var nodes = _nodeCount;
        var chunkSize = Std.int(rows / nodes);
        var threads = new Array<Thread<Int>>();

        for (n in 0...nodes) {
            var lo = n * chunkSize;
            var hi = (n == nodes - 1) ? rows : (n + 1) * chunkSize;
            var node = n;
            var f = fn;
            var t = Thread.spawn(function():Int {
                NumaTopology.bindCurrent(node);
                f(lo, hi, node);
                return 0;
            });
            threads.push(t);
        }

        for (t in threads) {
            t.join();
        }
    }

    private static var _instance:NumaPool;
    /** Process-wide singleton pool, lazily constructed on first call. */
    public static function global():NumaPool {
        if (_instance == null) _instance = new NumaPool();
        return _instance;
    }
}
