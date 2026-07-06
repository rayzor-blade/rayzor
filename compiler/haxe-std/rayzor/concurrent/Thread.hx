package rayzor.concurrent;

/**
 * Lightweight thread implementation similar to goroutines.
 *
 * Threads provide safe concurrent execution with compile-time Send/Sync validation.
 * All captured variables in the closure passed to spawn() must implement the Send trait.
 *
 * Example:
 * ```haxe
 * @:derive([Send])
 * class Message {
 *     public var data: String;
 *     public function new(data: String) { this.data = data; }
 * }
 *
 * var msg = new Message("Hello");
 * var handle = Thread.spawn(() -> {
 *     trace(msg.data); // OK: Message is Send
 *     return 42;
 * });
 *
 * var result = handle.join();
 * trace(result); // 42
 * ```
 */
@:native("rayzor::concurrent::Thread")
extern class Thread<T> {
    /**
     * Spawn a new thread to execute the given closure.
     *
     * All captured variables must implement the Send trait.
     * The closure itself is moved into the new thread.
     *
     * @param fn The closure to execute in the new thread
     * @return A handle to the spawned thread
     */
    @:native("spawn")
    public static function spawn<T>(fn: Void -> T): Thread<T>;

    /**
     * Wait for the thread to complete and return its result.
     *
     * This blocks the current thread until the spawned thread finishes.
     * The result type T must implement Send.
     *
     * @return The result produced by the thread's closure
     */
    @:native("join")
    public function join(): T;

    /**
     * Check if the thread has finished execution.
     *
     * Non-blocking check for thread completion.
     *
     * @return true if the thread has finished, false otherwise
     */
    @:native("is_finished")
    public function isFinished(): Bool;

    /**
     * Yield execution to allow other threads to run.
     *
     * This is a hint to the scheduler that the current thread
     * can pause to let other threads make progress.
     */
    @:native("yield_now")
    public static function yieldNow(): Void;

    /**
     * CPU spin-wait relax hint: PAUSE (x86) / YIELD (aarch64). Unlike
     * `yieldNow` this does NOT transition to the scheduler — the thread
     * stays runnable but the core drops power for one spin iteration.
     * Call it every iteration of a busy-wait: on x86 a bare spin loop
     * runs the core at full power (and triggers memory-order machine
     * clears), which is the dominant thermal cost on constrained parts.
     */
    @:native("cpu_relax")
    public static function cpuRelax(): Void;

    /**
     * Sleep the current thread for the specified duration in milliseconds.
     *
     * @param millis Duration to sleep in milliseconds
     */
    @:native("sleep")
    public static function sleep(millis: Int): Void;

    /**
     * Get the current thread's ID.
     *
     * @return Unique identifier for the current thread
     */
    @:native("current_id")
    public static function currentId(): Int;
}
