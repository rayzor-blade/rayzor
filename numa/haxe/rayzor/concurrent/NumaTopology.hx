package rayzor.concurrent;

/**
 * NUMA topology query + thread affinity binding.
 *
 * Thin wrapper around the `rayzor-numa` plugin's C-ABI surface. All methods
 * are static — there's one topology per process, queried lazily on first
 * call inside the plugin.
 *
 * On platforms without true NUMA (macOS, single-socket Linux, wasm,
 * Windows with one node) `nodeCount()` returns `1`, every CPU maps to
 * node `0`, and `bindCurrent(0)` succeeds as a soft affinity hint. The
 * same Haxe source therefore runs unchanged across multi-socket Linux
 * servers and the browser.
 *
 * Most callers should use the higher-level `NumaPool` for parallel-for
 * style work — this class is the low-level primitive `NumaPool` is built
 * on, and is exposed for advanced callers who need fine-grained control.
 *
 * Example:
 * ```haxe
 * trace(NumaTopology.available());     // true on dual-socket Linux, false on Mac
 * trace(NumaTopology.nodeCount());     // 2 or 4 on big servers, 1 elsewhere
 * trace(NumaTopology.cpuCount());      // total logical CPUs (always >= 1)
 *
 * // Pin a worker thread to a specific node:
 * Thread.spawn(() -> {
 *     NumaTopology.bindCurrent(1);
 *     // ... work on data co-located with node 1's memory controller
 * });
 * ```
 */
extern class NumaTopology {
    /**
     * `true` iff the runtime discovered a multi-node NUMA topology.
     *
     * `NumaPool` uses this to short-circuit affinity calls on platforms
     * where they're no-ops.
     */
    public static function available():Bool;

    /**
     * Number of NUMA nodes the runtime knows about. Always `>= 1`.
     *
     * On no-NUMA platforms this is `1` and every CPU lives on node `0`.
     */
    public static function nodeCount():Int;

    /** Total logical CPU count. Always `>= 1`. */
    public static function cpuCount():Int;

    /**
     * Which NUMA node a given logical CPU belongs to.
     *
     * Returns `0` on no-NUMA platforms, `-1` if `cpu` is out of range
     * (`cpu < 0 || cpu >= cpuCount()`).
     */
    public static function cpuToNode(cpu:Int):Int;

    /**
     * Pin the calling thread to all CPUs on `node`.
     *
     * Returns:
     * - `0` on success (including no-NUMA platforms where the call is a no-op).
     * - `-1` if the platform doesn't support thread affinity.
     * - `-2` if `node` is out of range.
     */
    public static function bindCurrent(node:Int):Int;

    /**
     * Clear any affinity hint on the calling thread (let it run anywhere).
     *
     * Returns `0` on success, `-1` if the platform doesn't support unbinding.
     */
    public static function unbindCurrent():Int;
}
