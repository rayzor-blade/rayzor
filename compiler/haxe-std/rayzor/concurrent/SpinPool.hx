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
 * through a lock-free protocol, so a dispatch costs a few atomic stores.
 *
 * Work distribution is CHUNK-STEALING, not static bands: workers (and the
 * caller) claim `chunk`-sized row ranges from a shared atomic cursor until
 * the range is exhausted. Static equal bands lose the join to the slowest
 * core (an E-core band runs 3-4x longer than a P-core band and the wall is
 * max-across-workers); stealing self-balances heterogeneous cores without
 * needing to know the topology. Each row is still computed by exactly one
 * worker with an unchanged per-row reduction order, so results remain
 * bit-identical to the serial loop.
 *
 * Protocol (per dispatch): the caller publishes the band closure, resets
 * the cursor and row/chunk cells, sets `pending = n-1`, then flips each
 * worker's state cell to 1 (the SC store publishes everything before it)
 * and unparks it. Workers and caller then race the cursor; a worker that
 * exhausts the cursor resets its state FIRST and then decrements `pending`
 * (that order is load-bearing: reversed, the caller could re-dispatch into
 * the slot and the worker's late reset would eat the new wakeup).
 *
 * Constraints:
 * - Hold the pool in an INSTANCE field / local and pass it down; do not
 *   cache it in a static.
 * - `shutdown()` MUST be called before the program exits: workers are
 *   never joined during normal operation, and the runtime waits for all
 *   live threads before tearing down JIT code.
 * - Captured state in the band closure must be read-only-shared or
 *   write-disjoint by row (as in `WorkerPool.parallelRows`).
 * - Idle workers spin briefly then PARK (Parker.park). The spin budget
 *   covers intra-token dispatch gaps so back-to-back kernel dispatches
 *   don't pay a park/unpark syscall per matmul.
 */
class SpinPool {
    /** Idle spins before parking. Default covers intra-token FFI-glue gaps
        (~150us); raise via RAYZOR_HAXE_POOL_SPINS when the Rust FFI pool is
        small enough not to contend (see the measured tradeoff below). */
    var spinBudget:Int;
    /** Emit a CPU PAUSE/YIELD each idle spin. Default OFF: on M-series the
        per-spin hint costs more (call + response latency in the hot join)
        than it saves and there is no thermal pressure. On thermally-
        constrained x86 (small-chassis NUC) a bare spin loop runs the core
        at full power and machine-clears on the shared cell — enable via
        RAYZOR_HAXE_POOL_RELAX=1 to trade a little dispatch latency for much
        lower spin power / less throttling. -1 = unresolved. */
    var relaxHint:Int;
    var _n:Int;
    var _ctl:Bytes;
    var _fn:(Int, Int, Int) -> Void;
    var _threads:Array<Thread<Int>>;
    var _alive:Bool;
    var _scratch0:Null<Bytes>;
    var _scratch1:Null<Bytes>;
    var _scratch2:Null<Bytes>;
    var _scratch0Cap:Int;
    var _scratch1Cap:Int;
    var _scratch2Cap:Int;
    var _profile:Int;

    /** Cell layout, one 128-byte line per cell (false-sharing stride):
        cell 0 = pending, 1 = shutdown, 2 = cursor, 3 = rows, 4 = chunk,
        5 = accumulated band wall (us), 6 = accumulated quantize wall (us),
        7 = dispatch count, 8 = inter-dispatch gap EWMA (us), 9 = last
        dispatch timestamp (us, wrapping), then per worker w in 1..n-1:
        state at 10+(w-1), pid at 10+(n-1)+(w-1). */
    inline function cell(i:Int):Atomic<Int> {
        var p:Ptr<Int> = Ptr.fromRaw(_ctl.address() + Usize.fromInt(i * 128));
        return Atomic.of(p);
    }

    inline function pending():Atomic<Int> return cell(0);

    inline function shut():Atomic<Int> return cell(1);

    inline function cursor():Atomic<Int> return cell(2);

    inline function rowsCell():Atomic<Int> return cell(3);

    inline function chunkCell():Atomic<Int> return cell(4);

    inline function bandUs():Atomic<Int> return cell(5);

    inline function quantUs():Atomic<Int> return cell(6);

    inline function dispatches():Atomic<Int> return cell(7);

    inline function gapEwmaUs():Atomic<Int> return cell(8);

    inline function lastDispatchUs():Atomic<Int> return cell(9);

    inline function stateOf(w:Int):Atomic<Int> return cell(10 + (w - 1));

    inline function pidOf(w:Int):Atomic<Int> return cell(10 + (_n - 1) + (w - 1));

    /** Spawn `n - 1` persistent workers (the caller is also a claimant). */
    public function new(n:Int) {
        if (n < 1) n = 1;
        _n = n;
        _threads = [];
        var cells = 10 + (n > 1 ? (n - 1) * 2 : 0);
        _ctl = Bytes.alloc(cells * 128);
        for (i in 0...cells) cell(i).store(0);
        _alive = true;
        _scratch0 = null;
        _scratch1 = null;
        _scratch2 = null;
        _scratch0Cap = 0;
        _scratch1Cap = 0;
        _scratch2Cap = 0;
        _profile = 0;
        spinBudget = 0; // resolved lazily on first dispatch (env reads in a
                        // constructor mis-resolve; see bugs_sys_getenv_in_ctor)
        relaxHint = -1; // resolved lazily alongside spinBudget
        CpuTopology.bindPerformance();
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

    public function profiling():Bool {
        if (_profile == 0) {
            var v = Sys.getEnv("RAYZOR_PROFILE_POOL");
            _profile = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _profile == 1;
    }

    /** Claim chunks from the shared cursor until exhausted. `who` is the
        claimant id (0 = caller) passed through as the band's node arg.
        Locals are explicitly Int-typed: the `if (n1 > rows) n1 = rows`
        reassign shape decays n1 to Float in MIR and the band call then
        fails LLVM verification (double passed to an i32 param). */
    inline function stealLoop(f:(Int, Int, Int) -> Void, who:Int):Void {
        var rows:Int = rowsCell().load();
        var chunk:Int = chunkCell().load();
        var cur = cursor();
        while (true) {
            var n0:Int = cur.fetchAdd(chunk);
            if (n0 >= rows) break;
            var n1:Int = (n0 + chunk > rows) ? rows : (n0 + chunk);
            f(n0, n1, who);
        }
    }

    function workerLoop(w:Int):Int {
        CpuTopology.bindPerformance();
        pidOf(w).store(Parker.registerParkable());
        // Hoist the atomic cells: cell() re-derives the control-block base
        // via an extern address() call per access, which is measurable in
        // the state/shut spin loops (profiled hot).
        var stateC = stateOf(w);
        var shutC = shut();
        var pendC = pending();
        var gapC = gapEwmaUs();
        var spins = 0;
        // relaxHint is resolved on the first dispatch; workers spawned before
        // it re-read on each idle entry via the field is too costly, so read
        // it once per idle transition below.
        var rlx = 0;
        while (true) {
            if (stateC.load() == 1) {
                var f = _fn;
                stealLoop(f, w);
                // Reset state BEFORE decrementing pending (see class doc).
                stateC.store(0);
                pendC.fetchAdd(-1);
                spins = 0;
                rlx = relaxHint; // refresh once per work cycle (post-init)
            } else {
                if (shutC.load() == 1) return 0;
                spins++;
                // PAUSE/YIELD each idle spin when enabled (x86 thermal; see
                // relaxHint). Gated: the hint's call cost regresses M-series.
                if (rlx == 1) Thread.cpuRelax();
                if (spins <= spinBudget) continue;
                // Idle beyond the tight-spin window: HOLD by yielding —
                // the worker stays runnable (no OS wake latency on the
                // next dispatch) but cedes its core to any thread with
                // real work (no starvation, degrades gracefully under
                // thermal throttling). Park only once we have been idle
                // well past the pool's own measured dispatch cadence.
                var holdUs:Int = gapC.load() * 4;
                if (holdUs < 200) holdUs = 200;
                if (holdUs > 20000) holdUs = 20000;
                var idleStart = Sys.time();
                var parked = false;
                var holdSpins = 0;
                while (stateC.load() != 1) {
                    if (shutC.load() == 1) return 0;
                    // Sys.time() + yieldNow() are SYSCALLS; issued per
                    // iteration they dominate idle cost (profiled: 52% of
                    // decode CPU landed in libsystem under workerLoop, and
                    // covering the gap with tight spin instead measured
                    // +32% tok/s). Relax-spin between checks and only pay
                    // the time+yield syscalls every 256 iterations.
                    holdSpins++;
                    if (rlx == 1) Thread.cpuRelax();
                    if ((holdSpins & 255) == 0) {
                        if ((Sys.time() - idleStart) * 1e6 > holdUs) {
                            parked = true;
                            break;
                        }
                        Thread.yieldNow();
                    }
                }
                if (parked) Parker.park();
                spins = 0;
            }
        }
    }

    /**
     * Distribute `rows` across the pool: `fn(lo, hi, node)` per claimed
     * chunk, with the caller claiming alongside the workers. Blocks until
     * every row is done. Small inputs run inline (dispatch overhead
     * exceeds the parallel win).
     */
    public function parallelRows(rows:Int, fn:(Int, Int, Int) -> Void):Void {
        if (rows <= 0) return;
        if (_n <= 1 || rows < 2 || !_alive) {
            fn(0, rows, 0);
            return;
        }
        if (spinBudget == 0) {
            // Defaults are DERIVED FROM THE PLATFORM (arch/thermal profile,
            // computed once in the runtime) — not hand-set constants. The
            // env vars only OVERRIDE for a specific box. Tight-spin window
            // before the yield-hold phase; see workerLoop.
            spinBudget = CpuTopology.poolSpinDefault();
            if (spinBudget <= 0) spinBudget = 2000;
            var sb = Sys.getEnv("RAYZOR_HAXE_POOL_SPINS");
            if (sb != null) {
                var v:Null<Int> = Std.parseInt(sb);
                if (v != null && v > 0) spinBudget = v;
            }
            relaxHint = CpuTopology.poolRelaxDefault();
            var rx = Sys.getEnv("RAYZOR_HAXE_POOL_RELAX");
            if (rx != null) {
                relaxHint = (rx != "0" && rx != "" && rx != "false") ? 1 : 0;
            }
        }
        // Record the inter-dispatch gap EWMA. Workers size their idle hold
        // from the pool's own cadence, so stale samples make wake/park timing
        // unstable across otherwise identical decode runs.
        var nowUs:Int = Std.int((Sys.time() * 1e6) % 2000000000.);
        var last:Int = lastDispatchUs().load();
        lastDispatchUs().store(nowUs);
        if (last != 0) {
            var gap:Int = nowUs - last;
            if (gap > 0 && gap < 10000000) {
                var e:Int = gapEwmaUs().load();
                var delta:Int = gap - e;
                gapEwmaUs().store(e + (delta >> 2));
            }
        }
        _fn = fn; // plain store; published by the SC state flips below
        // Chunk sizing: ~8 claims per worker amortizes cursor traffic while
        // leaving enough grains for stealing to absorb slow cores. Below
        // 4 rows/claimant we're in the fat-row regime (each row carries
        // large work, e.g. one attention kv-head band): claim single rows
        // so every claimant gets one — cursor traffic is noise there.
        var chunk:Int;
        if (rows < _n * 4) {
            chunk = 1;
        } else {
            chunk = Std.int(rows / (_n * 8));
            if (chunk < 8) chunk = 8;
        }
        cursor().store(0);
        rowsCell().store(rows);
        chunkCell().store(chunk);
        pending().store(_n - 1); // before ANY state flip
        var profile = profiling();
        var t0 = profile ? Sys.time() : 0.0;
        for (w in 1..._n) {
            stateOf(w).store(1);
            var pid = pidOf(w).load();
            if (pid != 0) Parker.unpark(pid);
        }
        stealLoop(fn, 0);
        var pendC = pending();
        // PAUSE/YIELD while the caller waits on the last stragglers when the
        // relax hint is enabled (x86 thermal). Default off: the per-iteration
        // call cost lengthens the hot join on M-series.
        if (relaxHint == 1) {
            while (pendC.load() > 0) { Thread.cpuRelax(); }
        } else {
            while (pendC.load() > 0) {}
        }
        if (profile) {
            bandUs().fetchAdd(Std.int((Sys.time() - t0) * 1e6));
            dispatches().fetchAdd(1);
        }
    }

    /** Accumulate externally-timed quantize wall (microseconds). */
    public function addQuantUs(us:Int):Void quantUs().fetchAdd(us);

    /** One-line profile of accumulated pool time; resets nothing. */
    public function profReport():String {
        return "band_ms=" + (bandUs().load() / 1000.0)
            + " quant_ms=" + (quantUs().load() / 1000.0)
            + " dispatches=" + dispatches().load();
    }

    /** Reusable per-pool scratch for hot kernels with sequential dispatches.
        The buffer may grow but never shrinks until shutdown, which avoids
        malloc/free churn in per-token quant matmuls and keeps Linux glibc
        arenas from ballooning on worker-heavy runs. */
    public function scratchBytes(slot:Int, size:Int):Bytes {
        if (size < 0) size = 0;
        if (slot == 0) {
            if (_scratch0 == null || _scratch0Cap < size) {
                if (_scratch0 != null) _scratch0.free();
                _scratch0 = Bytes.alloc(size);
                _scratch0Cap = size;
            }
            return cast _scratch0;
        }
        if (slot == 1) {
            if (_scratch1 == null || _scratch1Cap < size) {
                if (_scratch1 != null) _scratch1.free();
                _scratch1 = Bytes.alloc(size);
                _scratch1Cap = size;
            }
            return cast _scratch1;
        }
        if (_scratch2 == null || _scratch2Cap < size) {
            if (_scratch2 != null) _scratch2.free();
            _scratch2 = Bytes.alloc(size);
            _scratch2Cap = size;
        }
        return cast _scratch2;
    }

    function freeScratch():Void {
        if (_scratch0 != null) {
            _scratch0.free();
            _scratch0 = null;
            _scratch0Cap = 0;
        }
        if (_scratch1 != null) {
            _scratch1.free();
            _scratch1 = null;
            _scratch1Cap = 0;
        }
        if (_scratch2 != null) {
            _scratch2.free();
            _scratch2 = null;
            _scratch2Cap = 0;
        }
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
        freeScratch();
        _ctl.free();
    }
}
