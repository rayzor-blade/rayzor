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
 * node `0`, and `bindCurrent(0)` succeeds as a soft affinity hint.
 *
 * Most callers should use the higher-level `NumaPool` for parallel-for
 * style work — this class is the low-level primitive `NumaPool` is built
 * on, and is exposed for advanced callers who need fine-grained control.
 */
extern class NumaTopology {
    /** True iff the runtime discovered a multi-node NUMA topology. */
    public static function available():Bool;

    /** Number of NUMA nodes the runtime knows about. Always >= 1. */
    public static function nodeCount():Int;

    /** Total logical CPU count. Always >= 1. */
    public static function cpuCount():Int;

    /** Which NUMA node a given logical CPU belongs to. Returns 0 on no-NUMA, -1 if cpu out of range. */
    public static function cpuToNode(cpu:Int):Int;

    /** Pin the calling thread to all CPUs on `node`. Returns 0=ok, -1=unsupported, -2=invalid. */
    public static function bindCurrent(node:Int):Int;

    /** Clear any affinity hint on the calling thread. Returns 0=ok, -1=unsupported. */
    public static function unbindCurrent():Int;
}
