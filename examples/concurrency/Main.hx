import rayzor.concurrent.Thread;
import rayzor.concurrent.Arc;
import rayzor.concurrent.Mutex;
import rayzor.Box;
import rayzor.Atomic;

/**
 * Parallelism patterns in pure Haxe — identical behavior on native and wasm.
 *
 * These mirror the data-parallel shapes nue's Rust hot paths use
 * (`worker_pool.parallel_rows` band decomposition, work-stealing cursors,
 * atomic join counters, reductions). The point of the atomics + Arc/Mutex +
 * closure-capture work is that the SAME Haxe source runs the same way on
 * `rayzor run` (native threads) and `rayzor run --wasm` (in-guest threads over
 * shared memory) — so a kernel can live in portable Haxe instead of an FFI hop
 * into the Rust runtime for every parallel section.
 *
 *   rayzor run examples/concurrency/Main.hx
 *   rayzor run --wasm examples/concurrency/Main.hx
 *
 * Both targets print PASS for all three patterns.
 *
 * Known limitation (worker closures today):
 *   - Haxe `Array` operations are not safe inside a spawned worker (alloc /
 *     push / returning an Array across `join` are unreliable). Workers should
 *     exchange primitives and mutate shared state through Arc/Mutex/Atomic.
 *     The disjoint-buffer write form of `parallel_rows` therefore still goes
 *     through the Rust runtime; the banded-reduction form below is pure Haxe.
 */
class Main {
    static final WORKERS = 4;

    static function main() {
        Sys.println("=== Haxe parallelism patterns (native + wasm) ===");
        report("1. banded fork-join reduce  (parallel_rows: disjoint band -> partial -> fold)", bandedReduce());
        report("2. Arc<Mutex> shared state  (cross-worker mutual-exclusion accumulate)", arcMutexReduce());
        report("3. Atomic work-stealing     (lock-free shared cursor, each index once)", atomicWorkStealing());
        Sys.println("=== done ===");
    }

    static function report(name:String, pass:Bool) {
        Sys.println((pass ? "[PASS] " : "[FAIL] ") + name);
    }

    // --- 1. Banded fork-join with reduction -------------------------------
    // The parallel_rows shape: the index range is split into WORKERS disjoint
    // bands, each worker reduces its band independently and returns a partial,
    // the joiner folds the partials. (Deterministic fold order.)
    static function bandedReduce():Bool {
        final N = 1000000;
        final per = Std.int(N / WORKERS);
        var hs = new Array<Thread<Int>>();
        for (t in 0...WORKERS) {
            var lo = t * per;
            var hi = (t == WORKERS - 1) ? N : (t + 1) * per;
            hs.push(Thread.spawn(() -> {
                var partial = 0;
                for (i in lo...hi) partial += i;
                return partial;
            }));
        }
        var total = 0;
        for (h in hs) total += h.join();
        var expect = 0;
        for (i in 0...N) expect += i;
        return total == expect;
    }

    // --- 2. Arc<Mutex<T>> shared mutable accumulator ----------------------
    // Every worker locks the same guarded cell and folds its contribution in.
    // Real mutual exclusion on both targets: parking_lot on native, an in-guest
    // 3-state futex (memory.atomic.wait32/notify over shared memory) on wasm.
    static function arcMutexReduce():Bool {
        final per = 100000;
        var acc = new Arc(new Mutex(new Counter()));
        var hs = new Array<Thread<Int>>();
        for (t in 0...WORKERS) {
            hs.push(Thread.spawn(() -> {
                for (i in 0...per) {
                    var g = acc.lock();
                    var c = g.get();
                    c.v = c.v + 1;
                    g.unlock();
                }
                return 0;
            }));
        }
        for (h in hs) h.join();
        var g = acc.lock();
        var got = g.get().v;
        g.unlock();
        return got == WORKERS * per;
    }

    // --- 3. Atomic work-stealing cursor (lock-free dynamic balance) -------
    // nue analog: `parallel_rows_stealing` / the FLASH_DONE join counter.
    // Workers atomically claim the next index from one shared cell; fetchAdd is
    // a single i32.atomic.rmw on the shared memory, so every index is handed
    // out exactly once and the summed claims equal the full range.
    static function atomicWorkStealing():Bool {
        final total = 50000;
        var cell = Box.init(0);
        var cursor = Atomic.of(cell.asPtr()); // inferred Atomic<Int>; crosses threads, no annotation
        var hs = new Array<Thread<Int>>();
        for (t in 0...WORKERS) {
            hs.push(Thread.spawn(() -> {
                var localSum = 0;
                while (true) {
                    var item = cursor.fetchAdd(1); // claim next index, lock-free
                    if (item >= total) break;
                    localSum += item;
                }
                return localSum;
            }));
        }
        var sum = 0;
        for (h in hs) sum += h.join();
        var expect = 0;
        for (i in 0...total) expect += i;
        return sum == expect;
    }
}

@:derive([Send, Sync])
class Counter {
    public var v:Int;
    public function new() {
        this.v = 0;
    }
}
