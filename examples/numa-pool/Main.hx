import rayzor.concurrent.NumaTopology;
import rayzor.concurrent.NumaPool;

/**
 * NumaPool demo — exercises NUMA topology query + a small parallelFor.
 *
 * On macOS / single-socket Linux / WASM: `nodeCount()` is 1, so
 * parallelFor runs inline on the calling thread (no fanout).
 *
 * On multi-socket Linux: `nodeCount()` is 2+, and parallelFor spawns
 * one worker per NUMA node, each pinned via NumaTopology.bindCurrent
 * before invoking the closure.
 *
 * Run:
 *   rayzor run --rpkg ../../numa/rayzor-numa.rpkg Main.hx
 */
class Main {
    static function main() {
        trace("=== NumaPool Demo ===");

        // Topology query — works on every platform, even when NUMA is absent.
        var available = NumaTopology.available();
        var nodes = NumaTopology.nodeCount();
        var cpus = NumaTopology.cpuCount();
        trace("NUMA available: " + available);
        trace("Nodes: " + nodes);
        trace("CPUs:  " + cpus);

        // Show CPU → node mapping for the first 8 CPUs.
        var sampleCount = if (cpus < 8) cpus else 8;
        for (cpu in 0...sampleCount) {
            trace("  cpu " + cpu + " → node " + NumaTopology.cpuToNode(cpu));
        }

        // Drive a small parallelFor — sums [0, 100). On single-node systems
        // this runs inline; on multi-node it fans out across nodes.
        // We deliberately keep the work tiny so the demo finishes quickly;
        // real ML kernels feed parallelFor millions of indices.
        var pool = NumaPool.global();
        trace("pool nodeCount: " + pool.nodeCount());

        pool.parallelFor(8, function(idx:Int, node:Int):Void {
            trace("  parallelFor[" + idx + "] on node " + node);
        });

        // Multi-node fanout path runs even on single-NUMA-node hardware
        // when we force it — 4 Thread.spawn workers, each pinned via
        // NumaTopology.bindCurrent, processing 4 chunks of the range.
        trace("--- forced 4-node fanout ---");
        var fanout = NumaPool.withForcedNodes(4);
        fanout.parallelFor(16, function(idx:Int, node:Int):Void {
            trace("  fanout[" + idx + "] on node " + node);
        });

        trace("=== done ===");
    }
}
