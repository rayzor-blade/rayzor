package rayzor.concurrent;

/**
 * Mutual exclusion lock for protecting shared mutable data.
 *
 * Mutex provides exclusive access to the inner value T.
 * Typically used with Arc for shared mutable state across threads.
 *
 * Example:
 * ```haxe
 * class Counter {
 *     public var value: Int;
 *     public function new() { this.value = 0; }
 * }
 *
 * var counter = new Arc(new Mutex(new Counter()));
 * var counter_clone = counter.clone();
 *
 * Thread.spawn(() -> {
 *     var guard = counter_clone.get().lock();
 *     guard.get().value += 1;
 *     guard.unlock();
 * });
 * ```
 */
@:native("rayzor::concurrent::Mutex")
extern class Mutex<T> {
    /**
     * Create a new Mutex wrapping the given value.
     *
     * The Mutex starts unlocked.
     *
     * @param value The value to protect
     */
    public function new(value: T);

    /**
     * Create a new Mutex wrapping the given value.
     *
     * @deprecated Use `new Mutex(value)` instead
     */
    @:native("init")
    public static function init<T>(value: T): Mutex<T>;

    /**
     * Acquire the lock, blocking until available.
     *
     * Returns a guard that provides exclusive access to the inner value.
     * The lock is automatically released when the guard is dropped/unlocked.
     *
     * @return A lock guard providing exclusive access
     */
    @:native("lock")
    public function lock(): MutexGuard<T>;

    /**
     * Attempt to acquire the lock without blocking.
     *
     * @return A lock guard if successful, null if the lock is held
     */
    @:native("try_lock")
    public function tryLock(): Null<MutexGuard<T>>;

    /**
     * Check if the mutex is currently locked.
     *
     * This is a snapshot and may be immediately stale in concurrent contexts.
     *
     * @return true if locked
     */
    @:native("is_locked")
    public function isLocked(): Bool;
}

/**
 * Guard object providing exclusive access to a Mutex's inner value.
 *
 * The lock is held while the guard exists and is released when unlock() is called
 * or when the guard is dropped.
 */
@:native("rayzor::concurrent::MutexGuard")
@:autoDeref
extern class MutexGuard<T> {
    /**
     * Get a reference to the protected value.
     *
     * While holding the guard, you have exclusive mutable access.
     *
     * @return Reference to the inner value
     */
    @:native("get")
    public function get(): T;

    /**
     * Explicitly release the lock.
     *
     * After calling unlock(), the guard becomes invalid and get() will panic.
     */
    @:native("unlock")
    public function unlock(): Void;
}
