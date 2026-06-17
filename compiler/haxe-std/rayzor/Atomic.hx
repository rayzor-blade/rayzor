package rayzor;

/**
 * Atomic Int over a shared linear-memory cell (sequentially consistent).
 *
 * Each method lowers to one machine atomic instruction:
 * `i32.atomic.*` on wasm, Cranelift/LLVM atomics on native. There is no
 * function-call overhead and no lock.
 *
 * The wrapped `addr` MUST be 4-byte aligned or the access traps on wasm.
 *
 * Example:
 * ```haxe
 * var cell:Ptr<Int> = Ptr.alloc(1);
 * cell[0] = 0;
 * var a = Atomic.of(cell);
 * a.fetchAdd(1);          // returns old value (0), cell now 1
 * trace(a.load());        // 1
 * a.store(42);
 * a.compareExchange(42, 99); // swaps, returns 42
 * ```
 */
@:coreType
@:notNull
// Send: an Atomic is just a shared-memory address; sharing it across threads
// (so multiple workers atomic-rmw the SAME cell) is the entire point.
@:derive([Send])
@:native("rayzor::Atomic")
extern abstract Atomic {
    /** Wrap a 4-byte-aligned shared-memory address as an Atomic cell (identity). */
    @:native("of")
    public static function of(addr:Ptr<Int>):Atomic;

    /** Atomic load (SC). */
    @:native("load")
    public function load():Int;

    /** Atomic store (SC). */
    @:native("store")
    public function store(v:Int):Void;

    /** fetch_add (SC) — returns the OLD value. */
    @:native("fetchAdd")
    public function fetchAdd(v:Int):Int;

    /** compare-and-swap (SC) — returns the value READ (old). */
    @:native("compareExchange")
    public function compareExchange(expected:Int, replacement:Int):Int;
}
