package rayzor.concurrent;

/**
 * Atomically Reference Counted pointer for shared ownership across threads.
 *
 * Arc provides thread-safe shared ownership with automatic memory management.
 * The inner type T must implement Send + Sync to be shared between threads.
 *
 * Example:
 * ```haxe
 * @:derive([Send, Sync])
 * class SharedData {
 *     public var counter: Int;
 *     public function new() { this.counter = 0; }
 * }
 *
 * var data = new Arc(new SharedData());
 * var data_clone = data.clone(); // Increment ref count
 *
 * Thread.spawn(() -> {
 *     trace(data_clone.get().counter); // Shared access
 * });
 * ```
 */
@:native("rayzor::concurrent::Arc")
@:autoDeref
extern class Arc<T> {
    /**
     * Create a new Arc wrapping the given value (static factory method).
     *
     * The value type T must implement Send + Sync.
     *
     * @param value The value to wrap
     * @return A new Arc wrapping the value
     */
    @:native("init")
    public static function init<T>(value: T): Arc<T>;

    /**
     * Create a new Arc wrapping the given value.
     *
     * The value type T must implement Send + Sync.
     *
     * @param value The value to wrap
     */
    public function new(value: T);

    /**
     * Clone the Arc, incrementing the reference count.
     *
     * The cloned Arc points to the same underlying data.
     * Both Arc instances must be dropped for the data to be freed.
     *
     * @return A new Arc pointing to the same data
     */
    @:native("clone")
    public function clone(): Arc<T>;

    /**
     * Get a reference to the inner value.
     *
     * The value is shared and immutable unless wrapped in a Mutex.
     *
     * @return Reference to the inner value
     */
    @:native("get")
    public function get(): T;

    /**
     * Get the current reference count.
     *
     * This is primarily for debugging. The count may be inaccurate
     * due to concurrent clones/drops.
     *
     * @return Approximate reference count
     */
    @:native("strong_count")
    public function strongCount(): Int;

    /**
     * Try to unwrap the Arc and take ownership of the inner value.
     *
     * This only succeeds if this is the last Arc pointing to the data
     * (ref count = 1).
     *
     * @return The inner value if successful, null otherwise
     */
    @:native("try_unwrap")
    public function tryUnwrap(): Null<T>;

    /**
     * Get a pointer to the inner value for identity comparison.
     *
     * Two Arcs point to the same data if their pointers are equal.
     *
     * @return Pointer address as integer
     */
    @:native("as_ptr")
    public function asPtr(): Int;

    /**
     * Get a typed mutable pointer to the inner value.
     *
     * The returned Ptr<T> can be passed to C code via `.raw()`.
     * The Arc must outlive any use of the pointer.
     *
     * @return Typed pointer to the inner value
     */
    @:native("as_ptr_typed")
    public function asPtrTyped(): rayzor.Ptr<T>;

    /**
     * Get a typed read-only reference to the inner value.
     *
     * The returned Ref<T> can be passed to C code via `.raw()`.
     * The Arc must outlive any use of the reference.
     *
     * @return Typed read-only reference to the inner value
     */
    @:native("as_ref")
    public function asRef(): rayzor.Ref<T>;
}
