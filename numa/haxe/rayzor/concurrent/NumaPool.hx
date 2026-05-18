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
 * # v1 status (Phase 1b-ii)
 *
 * v1 ships the API surface + topology query + inline execution path. The
 * multi-node fanout that actually spawns `Thread.spawn` workers pinned via
 * `NumaTopology.bindCurrent` is **deferred** because of two blocking
 * compiler limitations:
 *
 * 1. `TypeKind::Function` is currently not `Send` in the trait checker
 *    ([trait_checker.rs:110-111](compiler/src/tast/trait_checker.rs#L110-L111)).
 *    Capturing the user's `fn:Int->Int->Void` into a `Thread.spawn`
 *    closure body would fail Send validation. A function value with no
 *    captures is bit-pattern Send by construction; loosening the rule
 *    (or adding `@:send`-on-Function inference) is a compiler change
 *    tracked separately.
 * 2. Parameterised multi-arg arrow `(idx:Int, node:Int)->Void` silently
 *    invalidates class registration — see
 *    `~/.claude/.../bugs_known.md`. The curried form `Int->Int->Void`
 *    used below is the workaround.
 *
 * The API surface is stable: future versions add fanout without changing
 * call sites. Today on every platform, `parallelFor` runs inline on the
 * calling thread; topology is queried correctly and `parallelRows`
 * partitions the range so the user closure body can take advantage of
 * the (node) argument when fanout eventually arrives.
 *
 * # Degenerate paths today
 *
 * - `nodeCount() == 1` (macOS, single-socket Linux, wasm, Windows with
 *   one node): no fanout possible anyway — inline execution is correct.
 * - All other platforms: still inline in v1 (see status note above).
 *
 * # Example
 *
 * ```haxe
 * import rayzor.concurrent.NumaPool;
 *
 * var pool = NumaPool.global();
 * trace("running on " + pool.nodeCount() + " nodes");
 *
 * pool.parallelFor(1000000, function(idx:Int, node:Int):Void {
 *     // ... do work; today runs on the calling thread,
 *     // tomorrow on a NUMA-pinned worker.
 * });
 * ```
 */
class NumaPool {
    private var _nodeCount:Int;

    public function new() {
        _nodeCount = NumaTopology.nodeCount();
    }

    /** Number of NUMA nodes this pool will fan out across (once fanout lands). */
    public function nodeCount():Int {
        return _nodeCount;
    }

    /**
     * Iterate `items` indices with the user closure, partitioned across
     * NUMA nodes.
     *
     * Uses the curried `Int->Int->Void` form (`fn(idx, node)`) because of
     * the multi-arg arrow class-registration bug — see class docs.
     */
    public function parallelFor(items:Int, fn:Int->Int->Void):Void {
        if (items <= 0) return;
        // v1: inline execution. fanout deferred — see class docs.
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
     * call per row — saves closure dispatch overhead and lets the inner
     * loop be tightly SIMD-optimised.
     */
    public function parallelRows(rows:Int, fn:Int->Int->Int->Void):Void {
        if (rows <= 0) return;
        // v1: one call covering [0, rows) on node 0. Once fanout lands,
        // this becomes one Thread.spawn per node with [lo, hi) per chunk.
        fn(0, rows, 0);
    }

    private static var _instance:NumaPool;
    /**
     * Process-wide singleton pool, lazily constructed on first call.
     *
     * Almost all callers want this — there's no benefit to multiple
     * pools per process and constructing fresh ones repeatedly burns
     * topology-query cycles (the OnceLock inside `rayzor-numa` already
     * amortises the syscalls, but the Haxe class itself still allocates).
     */
    public static function global():NumaPool {
        if (_instance == null) {
            _instance = new NumaPool();
        }
        return _instance;
    }
}
