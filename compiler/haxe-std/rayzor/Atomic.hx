package rayzor;

/**
 * Atomic cell over a shared linear-memory address (sequentially consistent).
 *
 * `Atomic<T>` is generic so the element type propagates from the wrapped
 * pointer at the call site — `Atomic.of(ptr:Ptr<Int>)` infers `Atomic<Int>` —
 * which keeps the value properly typed (and `Send`) when it crosses a thread
 * boundary, instead of degrading to `Dynamic`. `T` is a 32-bit cell type
 * (`Int`); each method lowers to one machine atomic instruction
 * (`i32.atomic.*` on wasm, Cranelift/LLVM atomics on native) — no call
 * overhead, no lock.
 *
 * The wrapped address MUST be 4-byte aligned or the access traps on wasm.
 *
 * Example:
 * ```haxe
 * var cell:Ptr<Int> = Ptr.alloc(1);
 * cell[0] = 0;
 * var a = Atomic.of(cell);   // a : Atomic<Int>
 * a.fetchAdd(1);             // returns old value (0), cell now 1
 * trace(a.load());           // 1
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
extern abstract Atomic<T> {
    /** Wrap a 4-byte-aligned shared-memory address as an Atomic cell (identity). */
    @:native("of")
    public static function of<T>(addr:Ptr<T>):Atomic<T>;

    /** Atomic load (SC). */
    @:native("load")
    public function load():T;

    /** Atomic store (SC). */
    @:native("store")
    public function store(v:T):Void;

    /** fetch_add (SC) — returns the OLD value. */
    @:native("fetchAdd")
    public function fetchAdd(v:T):T;

    /** compare-and-swap (SC) — returns the value READ (old). */
    @:native("compareExchange")
    public function compareExchange(expected:T, replacement:T):T;
}
