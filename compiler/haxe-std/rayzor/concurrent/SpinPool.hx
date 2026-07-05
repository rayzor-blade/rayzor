package rayzor.concurrent;

import rayzor.Atomic;
import rayzor.Bytes;
import rayzor.Ptr;
import rayzor.Usize;

/**
 * Persistent spin-wait worker pool for hot data-parallel kernels.
 *
 * `WorkerPool.parallelRows` spawns and joins fresh OS threads on every
 * call; for per-token kernels (quant matmul: hundreds of calls per token)
 * the thread create/teardown dominates and parallelism nets zero. A
 * `SpinPool` spawns its workers ONCE and re-dispatches work to them
 * through a lock-free generation protocol, so a dispatch costs a few
 * atomic stores instead of N pthread_create/join.
 *
 * Protocol (per dispatch): the caller publishes the band closure, sets
 * `pending = n-1`, then per worker stores (lo, hi, node) and flips that
 * worker's state cell to 1 — the sequentially-consistent state store
 * publishes everything written before it. The caller runs band 0 itself,
 * then spins until `pending == 0`. A worker spins on its state cell,
 * runs the band, resets state to 0 FIRST and then decrements `pending`
 * (that order is load-bearing: reversed, the caller could re-dispatch
 * into the slot and the worker's late reset would eat the new wakeup).
 *
 * Constraints:
 * - Hold the pool in an INSTANCE field / local and pass it down; do not
 *   cache it in a static.
 * - `shutdown()` MUST be called before the program exits: workers are
 *   never joined during normal operation, and the runtime waits for all
 *   live threads before tearing down JIT code.
 * - Captured state in the band closure must be read-only-shared or
 *   write-disjoint by band (as in `WorkerPool.parallelRows`).
 * - Idle workers spin briefly then PARK (Parker.park), so an idle pool
 *   costs little between dispatches.
 */
class SpinPool {
    var _n:Int;
    var _ctl:Bytes;
    var _fn:(Int, Int, Int) -> Void;
    var _threads:Array<Thread<Int>>;
    var _alive:Bool;

    /** Cell layout, one 128-byte line per cell (false-sharing stride):
        cell 0 = pending, cell 1 = shutdown,
        then per worker w in 1..n-1: base 2+(w-1)*4 = state, lo, hi, node. */
    inline function cell(i:Int):Atomic<Int> {
        var p:Ptr<Int> = Ptr.fromRaw(_ctl.address() + Usize.fromInt(i * 128));
        return Atomic.of(p);
    }

    inline function pending():Atomic<Int> return cell(0);

    inline function shut():Atomic<Int> return cell(1);

    inline function stateOf(w:Int):Atomic<Int> return cell(2 + (w - 1) * 4);

    inline function loOf(w:Int):Atomic<Int> return cell(2 + (w - 1) * 4 + 1);

    inline function hiOf(w:Int):Atomic<Int> return cell(2 + (w - 1) * 4 + 2);

    inline function nodeOf(w:Int):Atomic<Int> return cell(2 + (w - 1) * 4 + 3);

    inline function pidOf(w:Int):Atomic<Int> return cell(2 + (_n - 1) * 4 + (w - 1));

    /** Spawn `n - 1` persistent workers (the caller is band 0). */
    public function new(n:Int) {
        if (n < 1) n = 1;
        _n = n;
        _threads = [];
        var cells = 2 + (n > 1 ? (n - 1) * 5 : 0);
        _ctl = Bytes.alloc(cells * 128);
        for (i in 0...cells) cell(i).store(0);
        _alive = true;
        // Bind `this` to an explicit local: an implicit `this` capture in a
        // loop-spawned closure does not resolve.
        var self = this;
        for (w in 1...n) {
            var wid = w;
            var t = Thread.spawn(function():Int {
                return self.workerLoop(wid);
            });
            _threads.push(t);
        }
    }

    public function workers():Int return _n;

    function workerLoop(w:Int):Int {
        pidOf(w).store(Parker.registerParkable());
        var spins = 0;
        while (true) {
            if (stateOf(w).load() == 1) {
                var lo = loOf(w).load();
                var hi = hiOf(w).load();
                var node = nodeOf(w).load();
                var f = _fn;
                f(lo, hi, node);
                // Reset state BEFORE decrementing pending (see class doc).
                stateOf(w).store(0);
                pending().fetchAdd(-1);
                spins = 0;
            } else {
                if (shut().load() == 1) return 0;
                spins++;
                // Brief spin covers back-to-back dispatches; then park so an
                // idle pool costs nothing (no core burn between kernels, no
                // sleep-wake latency: the dispatcher unparks after flipping
                // state, and a pre-park unpark makes park return at once).
                if (spins > 2000) {
                    Parker.park();
                    spins = 0;
                }
            }
        }
    }

    /**
     * Band `rows` across the pool: `fn(lo, hi, band)` per band, band 0 on
     * the calling thread. Blocks until every band completes. Small inputs
     * run inline (dispatch overhead exceeds the parallel win).
     */
    public function parallelRows(rows:Int, fn:(Int, Int, Int) -> Void):Void {
        if (rows <= 0) return;
        if (_n <= 1 || rows < _n * 4 || !_alive) {
            fn(0, rows, 0);
            return;
        }
        _fn = fn; // plain store; published by the SC state flips below
        var chunk = Std.int(rows / _n);
        pending().store(_n - 1); // before ANY state flip
        for (w in 1..._n) {
            var lo = w * chunk;
            var hi = (w == _n - 1) ? rows : (w + 1) * chunk;
            loOf(w).store(lo);
            hiOf(w).store(hi);
            nodeOf(w).store(w);
            stateOf(w).store(1);
            var pid = pidOf(w).load();
            if (pid != 0) Parker.unpark(pid);
        }
        fn(0, chunk, 0);
        while (pending().load() > 0) {}
    }

    /** Stop and join all workers, then release the control block.
        Mandatory before process exit. Idempotent. */
    public function shutdown():Void {
        if (!_alive) return;
        _alive = false;
        shut().store(1);
        for (w in 1..._n) {
            var pid = pidOf(w).load();
            if (pid != 0) Parker.unpark(pid);
        }
        for (t in _threads) t.join();
        _threads = [];
        _ctl.free();
    }
}
