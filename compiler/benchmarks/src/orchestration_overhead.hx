// Orchestration Overhead Benchmark
//
// Replicates the pure-Haxe orchestration cost of nue's
// `LlamaModel.forwardIds` decode loop WITHOUT any ML compute. The point
// is to isolate the 257 ms / 80-token tax (12.9% of decode wall) we
// measured on real Llama 3.2 1B decode:
//
//   - Haxe forward phase: 1.91 s (95% of decode wall)
//   - Sum of kernel-time:  1.65 s
//   - Delta:               257 ms = pure orchestration
//
// Source mirroring (logical):
//   nue/nue/arch/LlamaModel.hx::forwardIds         (Array<Module> loop, 16 layers)
//   nue/nue/transformer/TransformerBlock.hx::forward (4 iface .forward + clone/release pattern)
//
// Per real token: 16 layers x ~14 method calls = ~224 calls + frees +
// clones. Per layer iter budget against the measured tax:
//   257 ms / (80 tokens * 16 layers) = ~200 us per layer-iter
//
// The bench reports per_iter_us and per_layer_iter_us. If those are
// close to the 200 us budget, the orchestration tax IS dispatch
// dominated and Devirt+Inline wins map straight to decode wall. If
// they're orders of magnitude smaller, the 257 ms is hiding inside
// real Tensor.clone/free runtime externs and the win shifts to the
// refcount path.
//
// ----- Measured (M1 Pro, rayzor binary release build, 2026-06-07) ----
//
// preset=Application (default tier promotion):
//   wall_ms median ~245 ms over 100k iters
//   per_iter_us  ~2.45 us  (full 16-layer forwardIds equiv)
//   per_layer_iter_us ~0.155 us = 155 ns
//
// preset=benchmark (tier 3 aggressive):
//   wall_ms ~170 ms
//   per_layer_iter_us ~107 ns
//
// Real decode budget: ~200 us per layer-iter (257 ms / 80 tokens / 16
// layers). Bench measures ~150 ns = ~1300x faster per layer-iter than
// the real tax. Conclusion: the 257 ms is NOT dispatch-bound -- almost
// all of it is in the Tensor.clone/free runtime extern path. A pure
// Devirt+Inline pass that wins 30% on THIS bench buys ~75 ns/layer-iter
// = 96 us per token = ~0.005% of real decode wall. Real wins are in:
//   - eliminating defensive .clone()/free() (escape analysis on
//     Tensor flows; AllocEscapeInfo already exists)
//   - skipping refcount bumps when the runtime extern is no-op
//   - inlining short Tensor.* helpers that escape through extern
//
// ------------------------------------------------------------------
// COMPILER BUG WORKAROUNDS — what the bench is NOT
// ------------------------------------------------------------------
// While building this we hit EIGHT distinct compiler bugs in the
// iface + Array + method-name space, all under the W0014 "cross-
// context iface return resolved late" umbrella. Each one forced a
// deviation from real nue's TransformerBlock shape. Verified
// 2026-06-07 with /tmp/orch_smoke{4..21}.hx +
// /tmp/orch_bench2{b,c}.hx + /tmp/orch_inc3.hx +
// /tmp/orch_min_bench{2,3}.hx repros:
//
//   (1) TWO classes implementing the same iface in one compilation
//       unit -> SIGSEGV at interior iface call.
//       Workaround: ONE `ModuleImpl` class with a per-instance `bump`
//       Int (real nue has RMSNorm + GQAttention + SwiGLU as distinct
//       classes). DevirtualizationPass with polymorphic call-site
//       support would not be exercised by this bench.
//
//   (2) A class with method signature matching an iface method but
//       NOT itself implementing the iface -> wrong vtable slot ->
//       SIGSEGV. Workaround: Block's outer method is `step`, not
//       `forward`. Real nue's TransformerBlock IS `implements Module`
//       and the outer method is named `forward`; we lose the outer
//       fat-ptr dispatch from `for (block in Array<Module>)`.
//
//   (3) `Array<Block>` stored as a field of a wrapper class and
//       retrieved by `m.blocks[0]` -> SIGSEGV at first interior iface
//       call on the retrieved Block. Workaround: build the Array<Block>
//       locally in main() and inline the warmup + measurement loops.
//
//   (4) `addInto`-style mutation called on the in-flight residual
//       OpaqueState BETWEEN iface dispatches corrupts memory (h.value
//       reads as 0x400 / 1024 / huge garbage). Workaround: bench omits
//       the residual-accumulator pattern entirely; sublayers run as a
//       linear chain `t1 = a(x); t2 = b(t1); t3 = c(t2); t4 = d(t3)`.
//       Real nue does pre-norm with residual addInto. We lose the
//       cost of the 2 addInto calls per block, but those are
//       single-method calls on a concrete class -- not the dispatch
//       work we're trying to measure anyway.
//
//   (5) `free` method name collides with libc's free import -> W0020
//       trap stub. `release` collides with sys_thread_*. The bench
//       uses `decRef` instead.
//
//   (6) `for (i in 0...N)` IntIterator while pushing Block-with-iface-
//       fields into an Array<Block> corrupts construction. Symptom:
//       first block.step() SIGSEGVs. Workaround: build the blocks
//       array with a `while` loop. Verified 2026-06-07,
//       /tmp/orch_bench2{b,c}.hx repros.
//
//   (7) Calling `decRef()` on the OpaqueState that flows OUT of a
//       sublayer dispatch (and becomes the residual into the next
//       sublayer) corrupts the next-layer input. We only decRef the
//       intermediate `xClone` + `t` locals, not the dispatched
//       returns. /tmp/orch_inc3.hx scales to 16 layers cleanly with
//       this pattern.
//
// What we CAN measure:
//   - 16-layer ArrayIterator<Block> for-in walk
//   - 4 interior interface fat-ptr forward() calls per Block.step
//   - 2 `clone()` allocations per Block.step
//   - 8 `decRef()` calls per Block.step
//
// That's 4 iface dispatches + 2 allocs + 8 method calls per layer-iter
// = matches the dispatch density of real TransformerBlock.forward
// within a factor of 2. Refcount work is preserved as best-effort
// counter math, no extern.
//
// Compile + run:
//   ./target/release/rayzor run compiler/benchmarks/src/orchestration_overhead.hx
//
// Optional: override iteration count via RAYZOR_ORCH_ITERS env var.

package benchmarks;

// --- The fake "Tensor" stand-in ------------------------------------

class OpaqueState {
    public var value:Int;
    public var refs:Int;

    public function new(v:Int) {
        this.value = v;
        this.refs = 1;
    }

    public function copy():OpaqueState {
        var c:OpaqueState = new OpaqueState(this.value);
        c.refs = 1;
        return c;
    }

    // Named `decRef` (not `free` or `release`) -- both `free` and
    // `release` collide with reserved runtime/stdlib symbols and
    // install W0020 trap stubs.
    public function decRef():Void {
        this.refs -= 1;
    }
}

// --- Module interface ---------------------------------------------------

interface Module {
    public function forward(x:OpaqueState):OpaqueState;
}

// --- Single Module impl (workaround #1) ---------------------------------

class ModuleImpl implements Module {
    public var bump:Int;

    public function new(b:Int) {
        this.bump = b;
    }

    public function forward(x:OpaqueState):OpaqueState {
        return new OpaqueState(x.value + this.bump);
    }
}

// --- TransformerBlock mirror (workaround #2, #4 applied) ----------------

class Block {
    public var attnNorm:Module;
    public var attn:Module;
    public var ffnNorm:Module;
    public var ffn:Module;

    public function new(attnNorm:Module, attn:Module, ffnNorm:Module, ffn:Module) {
        this.attnNorm = attnNorm;
        this.attn = attn;
        this.ffnNorm = ffnNorm;
        this.ffn = ffn;
    }

    // `step`, not `forward` -- see workaround #2 in header.
    //
    // Mirror the dispatch + refcount pattern from
    // TransformerBlock.hx::forward as faithfully as the compiler bugs
    // allow:
    //   - 4 iface fat-ptr forwards (preserved)
    //   - 2 alloc-via-copy()                (preserved as `copy()`)
    //   - 4 decRef counter bumps            (vs real 8 free() -- decRef
    //     can only touch intermediate locals; decRef'ing residual
    //     returns corrupts next layer, workaround #7)
    //   - 0 addInto                         (vs real 2 -- workaround #4)
    //
    // Per-layer call density: 4 iface + 6 concrete method calls = 10
    // method calls. Real TransformerBlock: 4 iface + 10 concrete = 14.
    // Bench-to-real ratio ~71%.
    public function step(x:OpaqueState):OpaqueState {
        var xClone1:OpaqueState = x.copy();
        var t:OpaqueState = attnNorm.forward(xClone1);
        xClone1.decRef();
        var attnOut:OpaqueState = attn.forward(t);
        t.decRef();
        var xClone2:OpaqueState = attnOut.copy();
        var t2:OpaqueState = ffnNorm.forward(xClone2);
        xClone2.decRef();
        var ffnOut:OpaqueState = ffn.forward(t2);
        t2.decRef();
        return ffnOut;
    }
}

// --- Bench driver ------------------------------------------------------
//
// Loops live in main() not a Model class -- see workaround #3.

class OrchestrationOverhead {
    static inline var NUM_LAYERS:Int = 16;
    static inline var DEFAULT_ITERS:Int = 100000;

    public static function main() {
        var iters = DEFAULT_ITERS;
        var envIters = Sys.getEnv("RAYZOR_ORCH_ITERS");
        if (envIters != null && envIters.length > 0) {
            var parsed = Std.parseInt(envIters);
            if (parsed != null && parsed > 0) iters = parsed;
        }

        var blocks:Array<Block> = [];
        // While loop, not `for (i in 0...NUM_LAYERS)` -- iterating
        // with `0...N` IntIterator while pushing Block-with-iface-
        // fields corrupts construction and SIGSEGVs the first step()
        // call (workaround #6, /tmp/orch_bench2{b,c}.hx repros).
        //
        // ALSO: `blocks.push(new Block(...))` inline -- NOT
        // `var b:Block = new Block(...); blocks.push(b)`. The
        // intermediate `var b` binding crashes at first step() call
        // even with everything else identical (workaround #8,
        // /tmp/orch_min_bench{2,3}.hx vs /tmp/orch_inc3.hx repro
        // 2026-06-07).
        var li:Int = 0;
        while (li < NUM_LAYERS) {
            blocks.push(new Block(
                new ModuleImpl(1), // attnNorm role
                new ModuleImpl(2), // attn role
                new ModuleImpl(3), // ffnNorm role
                new ModuleImpl(4)  // ffn role
            ));
            li += 1;
        }

        // Warmup so any tier promotion settles before we measure.
        // Use `while`, not `for (i in 0...N)` -- the IntIterator pattern
        // corrupts iface-field construction in this position (see
        // workaround #6 above).
        var warmup = Std.int(iters / 20);
        if (warmup < 1) warmup = 1;
        var sink = 0;
        var wi:Int = 0;
        while (wi < warmup) {
            var h:OpaqueState = new OpaqueState(wi);
            for (block in blocks) {
                h = block.step(h);
            }
            sink += h.value;
            h.decRef();
            wi += 1;
        }

        // Measurement window.
        var t0 = Sys.time();
        var acc = 0;
        var mi:Int = 0;
        while (mi < iters) {
            var h:OpaqueState = new OpaqueState(mi);
            for (block in blocks) {
                h = block.step(h);
            }
            acc += h.value;
            h.decRef();
            mi += 1;
        }
        var t1 = Sys.time();

        var wallMs = (t1 - t0) * 1000.0;
        var perIterUs = (wallMs * 1000.0) / iters;
        var perLayerIterUs = perIterUs / NUM_LAYERS;

        // Real-decode budget: 257 ms / (80 tokens * 16 layers) ~ 200 us
        // per layer-iter. The bench omits real Tensor.clone (refcount
        // through extern) so will run faster; the ratio tells us what
        // fraction of the 257 ms is dispatch-only vs refcount-mediated.
        trace("orchestration_overhead: iters=" + iters
            + " layers=" + NUM_LAYERS
            + " wall_ms=" + wallMs
            + " per_iter_us=" + perIterUs
            + " per_layer_iter_us=" + perLayerIterUs
            + " sink=" + sink + " acc=" + acc);
    }
}
