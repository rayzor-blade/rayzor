package rayzor;

/**
 * Single-owner heap allocation.
 *
 * Box<T> allocates a value on the heap with exclusive ownership.
 * Unlike Arc<T>, there is no reference counting overhead.
 * The owner is responsible for calling free() when done.
 *
 * Example:
 * ```haxe
 * var boxed = Box.init(42);
 * var ptr = boxed.asPtr();    // borrow as Ptr<Int>
 * var val = boxed.unbox();    // take ownership
 * boxed.free();               // release memory
 * ```
 */
@:native("rayzor::Box")
extern abstract Box<T> {
    /** Allocate a value on the heap */
    @:native("init")
    public static function init<T>(value:T):Box<T>;

    /** Get the inner value (moves ownership out) */
    @:native("unbox")
    public function unbox():T;

    /** Get a mutable pointer to the inner value (borrow) */
    @:native("as_ptr")
    public function asPtr():Ptr<T>;

    /** Get a read-only reference to the inner value (borrow) */
    @:native("as_ref")
    public function asRef():Ref<T>;

    /** Get the raw heap address */
    @:native("raw")
    public function raw():Usize;

    /** Free the box and its contents */
    @:native("free")
    public function free():Void;
}
