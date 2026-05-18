// Phase 1a: every Tensor allocated by zeros/ones/full/fromArray/rand
// defaults to device tag 0 (CPU) and NUMA node -1 (no affinity).
// Views (reshape/transpose/slice/permute) inherit the parent's device.

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var a = Tensor.zeros([4, 4], F32);
        trace(a.deviceTag());          // 0 (CPU)
        trace(a.numaNode());           // -1 (no affinity)

        var b = Tensor.ones([2, 3], F32);
        trace(b.deviceTag());          // 0
        trace(b.numaNode());           // -1

        // View inherits device + numa_node from parent.
        var v = a.reshape([16]);
        trace(v.deviceTag());          // 0
        trace(v.numaNode());           // -1

        var t = a.transpose();
        trace(t.deviceTag());          // 0
        trace(t.numaNode());           // -1
    }
}
