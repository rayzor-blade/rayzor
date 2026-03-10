package rayzor.concurrent;

/**
 * Lazy future for async computation.
 *
 * A Future stores a closure but does NOT execute it until `.await()` or `.then()`
 * is called. This is tokio-style lazy evaluation.
 *
 * Two paths for consuming a Future:
 * - **Blocking**: `f.await()` — spawns the closure on a worker thread and blocks until resolved
 * - **Non-blocking**: `f.then(callback)` — spawns and calls `callback(result)` when done
 *
 * Example:
 * ```haxe
 * var f = Future.create(() -> {
 *     Thread.sleep(100);
 *     return 42;
 * });
 * trace("not blocked yet");
 * var result = f.await();  // spawns + blocks → 42
 * trace(result);
 *
 * // Non-blocking path
 * Future.create(() -> 99).then((v) -> trace(v));
 * ```
 */
@:native("rayzor::concurrent::Future")
extern class Future<T> {
    /**
     * Create a lazy future from a closure.
     *
     * The closure is NOT executed immediately — it is stored and only
     * runs when `.await()` or `.then()` is called.
     *
     * @param fn The closure to execute lazily
     * @return A lazy Future handle
     */
    @:native("create")
    public static function create<T>(fn:Void->T):Future<T>;

    /**
     * Await the future: spawn if pending, block until resolved, return value.
     *
     * - If the future hasn't started: spawns it on a worker thread, then blocks
     * - If already running: blocks until resolved
     * - If already resolved: returns the value immediately
     *
     * @return The result produced by the future's closure
     */
    @:native("await")
    public function await():T;

    /**
     * Register a callback to run when the future resolves (non-blocking).
     *
     * If the future hasn't started, this also spawns it. The callback is
     * called on the worker thread when the future resolves.
     *
     * @param callback Function to call with the result value
     */
    @:native("then")
    public function then(callback:T->Void):Void;

    /**
     * Poll the future (non-blocking).
     *
     * @return The result if resolved, null if still pending
     */
    @:native("poll")
    public function poll():Null<T>;

    /**
     * Check if the future has resolved.
     *
     * @return true if the result is available
     */
    @:native("isReady")
    public function isReady():Bool;

    /**
     * Wait for all outstanding futures (and threads) to complete.
     *
     * Blocks the calling thread until every spawned future and thread
     * has finished. Useful before program exit to ensure `.then()`
     * callbacks have completed.
     */
    @:native("join")
    public static function join():Void;

    /**
     * Await the future with a timeout in milliseconds.
     *
     * Same as `.await()` but returns null if the timeout expires
     * before the future resolves.
     *
     * @param millis Maximum time to wait in milliseconds
     * @return The result if resolved within timeout, null otherwise
     */
    @:native("awaitTimeout")
    public function awaitTimeout(millis:Int):Null<T>;

    /**
     * Cancel the future (cooperative cancellation).
     *
     * - If pending: cancels immediately, returns true
     * - If running: sets cancellation flag, returns true
     * - If resolved: returns false
     *
     * After cancellation, `.await()` returns null.
     *
     * @return true if cancellation was requested
     */
    @:native("cancel")
    public function cancel():Bool;

    /**
     * Check if the future has been cancelled.
     *
     * @return true if cancellation was requested
     */
    @:native("isCancelled")
    public function isCancelled():Bool;

    /**
     * Like JavaScript's `Promise.all()`: takes an array of futures,
     * spawns them all in parallel, and returns a new Future that
     * resolves to an Array of results.
     *
     * @param futures Array of Future handles
     * @return A Future that resolves to Array<T> when all complete
     */
    @:native("all")
    public static function all<T>(futures:Array<Future<T>>):Future<Array<T>>;

    /**
     * Like JavaScript's `Promise.race()`: takes an array of futures,
     * spawns them all in parallel, and returns a new Future that
     * resolves with the first result.
     *
     * @param futures Array of Future handles
     * @return A Future that resolves with the first completed result
     */
    @:native("race")
    public static function race<T>(futures:Array<Future<T>>):Future<T>;
}
