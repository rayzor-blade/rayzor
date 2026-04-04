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

        var guard = mtx.lock();
        trace("Mutex locked");

        if (mtx.isLocked()) trace("Is locked: true"); else trace("Is locked: false");

        guard.unlock();
        trace("Mutex unlocked");

        if (mtx.isLocked()) trace("Is locked: true"); else trace("Is locked: false");

        trace("=== Done ===");
    }
}
