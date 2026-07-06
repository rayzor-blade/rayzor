package rayzor.concurrent;

/**
 * CPU topology query + thread affinity binding.
 *
 * Thin wrapper around the runtime's topology surface. All methods are
 * static — there's one topology per process, queried lazily by the
 * native side on first call.
 *
 * NUMA-aware on multi-socket Linux + Windows servers (worker threads
 * pinned per node, memory landed via first-touch on the bound node).
 * On UMA hardware (M1/M2, single-socket Linux, Windows laptops, wasm)
 * `nodeCount()` returns `1`, every CPU maps to node `0`, and
 * `bindToNode(0)` succeeds as a soft affinity hint.
 *
 * Most callers should use the higher-level `WorkerPool` for parallel-
 * for style work — this class is the low-level primitive `WorkerPool`
 * is built on, and is exposed for callers who need fine-grained
 * control over pinning.
 */
extern class CpuTopology {
    /** True iff the runtime discovered a multi-node NUMA topology. */
    @:native("rayzor_topology_multi_node")
    public static function multiNode():Bool;

    /** Number of NUMA nodes the runtime knows about. Always >= 1. */
    @:native("rayzor_topology_node_count")
    public static function nodeCount():Int;

    /** Total logical CPU count. Always >= 1. */
    @:native("rayzor_topology_cpu_count")
    public static function cpuCount():Int;

    /** Physical performance-core count on hybrid (big.LITTLE) parts;
        equals `cpuCount()` where no hybrid split is exposed. */
    @:native("rayzor_topology_perf_core_count")
    public static function perfCoreCount():Int;

    /** Which NUMA node a given logical CPU belongs to. Returns 0 on
        no-NUMA, -1 if cpu out of range. */
    @:native("rayzor_topology_cpu_to_node")
    public static function cpuToNode(cpu:Int):Int;

    /** Pin the calling thread to all CPUs on `node`. Returns 0=ok,
        -1=unsupported, -2=invalid. */
    @:native("rayzor_topology_bind_to_node")
    public static function bindToNode(node:Int):Int;

    /** Clear any affinity hint on the calling thread. Returns 0=ok,
        -1=unsupported. */
    @:native("rayzor_topology_unbind")
    public static function unbind():Int;
}
