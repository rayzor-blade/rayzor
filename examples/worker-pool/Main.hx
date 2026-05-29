import rayzor.concurrent.CpuTopology;
import rayzor.concurrent.WorkerPool;

/**
 * WorkerPool demo — exercises CPU topology query + a small parallelFor.
 *
 * On macOS / single-socket Linux / WASM: `nodeCount()` is 1, so
 * parallelFor runs inline on the calling thread (no fanout).
 *
 * On multi-socket Linux: `nodeCount()` is 2+, and parallelFor spawns
 * one worker per NUMA node, each pinned via CpuTopology.bindToNode
 * before invoking the closure.
 *
 * Run:
 *   rayzor run Main.hx
 */
class Main {
    static function main() {
        trace("=== WorkerPool Demo ===");

        // Topology query — works on every platform.
        var multiNode = CpuTopology.multiNode();
        var nodes = CpuTopology.nodeCount();
        var cpus = CpuTopology.cpuCount();
        trace("multi-node NUMA: " + multiNode);
        trace("Nodes: " + nodes);
        trace("CPUs:  " + cpus);

        // Show CPU → node mapping for the first 8 CPUs.
        var sampleCount = if (cpus < 8) cpus else 8;
        for (cpu in 0...sampleCount) {
            trace("  cpu " + cpu + " → node " + CpuTopology.cpuToNode(cpu));
        }

        // Drive a small parallelFor. On single-node systems this runs
        // inline; on multi-node it fans out across nodes.
        var pool = WorkerPool.global();
        trace("pool nodeCount: " + pool.nodeCount());

        pool.parallelFor(8, function(idx:Int, node:Int):Void {
            trace("  parallelFor[" + idx + "] on node " + node);
        });

        // Forced fanout path runs even on single-NUMA-node hardware —
        // 4 Thread.spawn workers, each pinned via CpuTopology.bindToNode,
        // processing 4 chunks of the range.
        trace("--- forced 4-worker fanout ---");
        var fanout = WorkerPool.withForcedNodes(4);
        fanout.parallelFor(16, function(idx:Int, node:Int):Void {
            trace("  fanout[" + idx + "] on node " + node);
        });

        trace("=== done ===");
    }
}
