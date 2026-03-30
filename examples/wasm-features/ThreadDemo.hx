import rayzor.concurrent.Mutex;

/**
 * Concurrency demo — mutex synchronization.
 * Browser: Atomics on SharedArrayBuffer.
 *
 * Run: rayzor build --target wasm --browser examples/wasm-features/ThreadDemo.hx
 */
class ThreadDemo {
    static function main() {
        trace("=== Thread Demo ===");

        // Mutex
        var mtx = new Mutex(0);
        trace("Mutex created");

        mtx.lock();
        trace("Mutex locked");

        trace("Is locked: " + mtx.isLocked());

        mtx.unlock();
        trace("Mutex unlocked");

        trace("Is locked: " + mtx.isLocked());

        trace("=== Done ===");
    }
}
