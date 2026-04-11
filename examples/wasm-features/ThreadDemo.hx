import rayzor.concurrent.Thread;
import rayzor.concurrent.Mutex;

/**
 * Thread demo — real concurrent execution primitives.
 *
 * Exercises Thread.spawn / join / currentId / sleep alongside Mutex.
 * On wasmtime the worker pool is single-threaded (spawn tasks run
 * synchronously on join), but the public API is identical.
 *
 * Run: rayzor run --wasm examples/wasm-features/ThreadDemo.hx
 */
@:derive([Send])
class Counter {
    public var value:Int;
    public function new(v:Int) { this.value = v; }
}

class ThreadDemo {
    static function main() {
        trace("=== Thread Demo ===");

        // --- Thread.currentId ---
        var mainId = Thread.currentId();
        trace("main thread id = " + mainId);

        // --- Basic spawn + join with captured primitive ---
        var x = 21;
        var h1 = Thread.spawn(() -> {
            return x * 2;
        });
        trace("spawn(x*2) join = " + h1.join());

        // --- Spawn with captured class instance (Send-derived) ---
        var c = new Counter(100);
        var h2 = Thread.spawn(() -> {
            return c.value + 42;
        });
        trace("spawn(counter+42) join = " + h2.join());

        // --- Two more independent spawns ---
        var a = 7;
        var b = 5;
        var hA = Thread.spawn(() -> { return a * a; });
        var hB = Thread.spawn(() -> { return b * b; });
        trace("spawn(7^2) + spawn(5^2) = " + (hA.join() + hB.join()));

        // --- Mutex round-trip ---
        var mtx = new Mutex(0);
        var guard = mtx.lock();
        trace("mutex locked: " + mtx.isLocked());
        guard.unlock();
        trace("mutex unlocked: " + mtx.isLocked());

        // --- Thread.sleep (returns quickly, just testing it doesn't crash) ---
        Thread.sleep(1);
        trace("after sleep(1)");

        trace("=== Done ===");
    }
}
