package rayzor.concurrent;

/**
 * Worker pool for data-parallel CPU work.
 *
 * `WorkerPool` is the canonical entry point for CPU-parallel kernels in
 * rayzor / nue — anything that iterates over a large index range (matmul
 * rows, conv tiles, attention heads, elementwise sweeps, KV-cache fills,
 * etc.) should route through `WorkerPool.global().parallelFor(...)`.
 *
 * On multi-NUMA-node systems (multi-socket Linux / Windows servers) the
 * pool pins one worker per node via `CpuTopology.bindToNode(node)` so
 * memory allocations land first-touch on the bound node's controller.
 * On UMA hardware (M1/M2, single-socket Linux, Windows laptops, wasm)
 * `CpuTopology.nodeCount()` returns 1 and the pool either runs inline
 * on the calling thread or fans out via `withForcedNodes(N)`.
 *
 * Behaviour:
 * - **Multi-node**: spawns one worker per NUMA node, each pinned via
 *   `CpuTopology.bindToNode(node)` before invoking the user closure.
 * - **Single-node** (M1/M2, single-socket Linux, wasm, Windows laptops):
 *   no fanout — runs inline on the calling thread. Use
 *   `withForcedNodes(N)` to fan out anyway.
 * - **Small work** (`items < nodeCount * 2`): also inline.
 *
 * Example:
 * ```haxe
 * var pool = WorkerPool.global();
 * pool.parallelFor(1000000, function(idx:Int, node:Int):Void {
 *     // Each worker runs on a NUMA-pinned thread when nodeCount > 1.
 * });
 * ```
 */
class WorkerPool {
    private var _nodeCount:Int;

    public function new() {
        _nodeCount = CpuTopology.nodeCount();
    }

    /**
     * Construct a pool with a forced node count, ignoring the runtime
     * topology. Useful for fan-out on UMA hardware where the topology
     * reports 1 node but we want N workers. `bindToNode` becomes a
     * soft-affinity hint on no-NUMA platforms, but the `Thread.spawn`
     * dispatch is real.
     */
    public static function withForcedNodes(nodes:Int):WorkerPool {
        var p = new WorkerPool();
        if (nodes > 0) p._nodeCount = nodes;
        return p;
    }

    /** Number of worker slots this pool fans out across. */
    public function nodeCount():Int return _nodeCount;

    /**
     * Iterate `items` indices with the user closure `fn(idx, node)`,
     * partitioned across worker slots.
     */
    public function parallelFor(items:Int, fn:(idx:Int, node:Int)->Void):Void {
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
        // Only pin affinity on real multi-NUMA hardware. On UMA (macOS, wasm,
        // laptops, single-socket) bindToNode is a meaningless soft hint, and
        // calling it from several freshly-spawned worker threads has crashed
        // (native SIGILL / wasm host-import trap), so skip it entirely there.
        var multiNode = CpuTopology.multiNode();

        for (n in 0...nodes) {
            var lo = n * chunkSize;
            var hi = (n == nodes - 1) ? items : (n + 1) * chunkSize;
            var node = n;
            var f = fn;
            var t = Thread.spawn(function():Int {
                if (multiNode) CpuTopology.bindToNode(node);
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
     * Block-partitioned variant: invoke `fn(rowStart, rowEnd, node)`
     * once per worker with a contiguous half-open block range.
     */
    public function parallelRows(
        rows:Int,
        fn:(rowStart:Int, rowEnd:Int, node:Int)->Void
    ):Void {
        if (rows <= 0) return;

        if (_nodeCount <= 1 || rows < _nodeCount) {
            fn(0, rows, 0);
            return;
        }

        var nodes = _nodeCount;
        var chunkSize = Std.int(rows / nodes);
        var threads = new Array<Thread<Int>>();
        // See parallelFor: skip affinity binding on UMA (it's a no-op hint that
        // has crashed when issued from multiple fresh worker threads).
        var multiNode = CpuTopology.multiNode();

        for (n in 0...nodes) {
            var lo = n * chunkSize;
            var hi = (n == nodes - 1) ? rows : (n + 1) * chunkSize;
            var node = n;
            var f = fn;
            var t = Thread.spawn(function():Int {
                if (multiNode) CpuTopology.bindToNode(node);
                f(lo, hi, node);
                return 0;
            });
            threads.push(t);
        }

        for (t in threads) {
            t.join();
        }
    }

    private static var _instance:WorkerPool;
    /** Process-wide singleton pool, lazily constructed on first call. */
    public static function global():WorkerPool {
        if (_instance == null) _instance = new WorkerPool();
        return _instance;
    }
}
