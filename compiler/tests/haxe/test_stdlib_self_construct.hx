// Regression test: a stdlib class constructed from within its OWN static
// method (intra-import-module construction) must not crash.
//
// Before the fix: the monomorphizer minted specialized-function ids at a
// fixed start of 10_000 (stride 1), while the import-merge renumbered each
// import module by `100_000 + N*10_000` (stride 10_000). A monomorphized
// instance at id 10_000 in import module N therefore renumbered to
// `base(N)+10_000 == base(N+1)` — the slot-0 (the `malloc` extern) of the
// next import module — and the merge silently overwrote it. So
// `WorkerPool.withForcedNodes()`'s `new WorkerPool()` heap-alloc CallDirect
// (to WorkerPool's malloc, renumbered to e.g. 230000) dispatched into a
// BalancedTree `get_height__i64_i64` instance instead, returned a tree
// height (a small int) as the object pointer, and SIGSEGV'd (exit 139).
// `new WorkerPool()` from user code worked because the user module emits a
// call to its OWN low-id malloc, never the renumbered import one.
//
// After the fix: the monomorphizer seeds its id counter above the module's
// existing ids, keeping instance ids inside the module's own [0, stride)
// band so they never alias another import module's base after renumbering.
//
// WorkerPool is the concrete trigger; the bug was systemic (any import
// module whose slot-0 collided with a monomorphized instance of the
// previous import module).

import rayzor.concurrent.WorkerPool;

class Main {
    static function main() {
        // `withForcedNodes` does `new WorkerPool()` inside a stdlib static
        // method — the exact intra-import-module construction that crashed.
        var a = WorkerPool.withForcedNodes(3);
        // `global()` also constructs WorkerPool internally (via a static var).
        var g = WorkerPool.global();

        if (a.nodeCount() == 3 && g.nodeCount() >= 1) {
            Sys.println("PASS stdlib-self-construct forced=" + a.nodeCount() + " global=" + g.nodeCount());
        } else {
            Sys.println("FAIL forced=" + a.nodeCount() + " (want 3) global=" + g.nodeCount() + " (want >=1)");
        }
    }
}
