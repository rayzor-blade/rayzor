import rayzor.concurrent.Thread;
import rayzor.concurrent.Future;
import rayzor.concurrent.Mutex;
import rayzor.concurrent.Channel;

/**
 * Concurrency demo — threads, futures, mutex, channels.
 * Browser: Web Workers + SharedArrayBuffer + Atomics.
 * WASI: native threads.
 *
 * Run: rayzor run --wasm examples/wasm-features/ThreadDemo.hx
 * Browser: rayzor build --target wasm --browser examples/wasm-features/ThreadDemo.hx
 */
class ThreadDemo {
    static function main() {
        trace("=== Thread Demo ===");

        // Future — lazy async computation
        var f = Future.create(function():Int {
            return 21 * 2;
        });
        trace("Future created");
        trace("Is ready: " + f.isReady());

        // Mutex
        var mtx = new Mutex(0);
        trace("Mutex created");
        mtx.lock();
        trace("Mutex locked");
        mtx.unlock();
        trace("Mutex unlocked");

        // Channel
        var ch = new Channel();
        trace("Channel created, empty=" + ch.isEmpty());

        trace("=== Done ===");
    }
}
