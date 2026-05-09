package rayzor.concurrent;

/**
 * Multi-channel non-deterministic receive.
 *
 * Polls an array of `Channel<T>` and returns the first channel that yields
 * a value, mimicking Go's `select { case <-ch1; case <-ch2; ... }`.
 *
 * Two flavours:
 * - `Select.recv(channels)` blocks until any channel has a value.
 * - `Select.tryRecv(channels)` returns immediately; `result.index == -1`
 *   when no channel was ready.
 *
 * Example:
 * ```haxe
 * var ch1 = new Channel<Msg>(0);
 * var ch2 = new Channel<Msg>(0);
 *
 * Thread.spawn(() -> ch2.send(new Msg(42)));
 *
 * var r = Select.recv([ch1, ch2]);
 * if (r.index == 0) trace("from ch1: " + r.value.v);
 * else if (r.index == 1) trace("from ch2: " + r.value.v);  // 42
 * ```
 *
 * Closed empty channels yield `index = <their slot>` and `value = null`,
 * matching Go's "receive on closed channel returns zero value" semantics —
 * use this to detect closure inside a select arm.
 *
 * Channels in a single Select call must share a common element type `T`.
 * For heterogeneous selects, declare `Channel<Dynamic>` and cast.
 */
@:native("rayzor::concurrent::Select")
extern class Select {
    /** Block until any channel has a value. Returns its `(index, value)`. */
    @:native("rayzor_select_recv")
    public static function recv<T>(channels:Array<Channel<T>>):SelectResult<T>;

    /**
     * Poll once across all channels. Returns immediately.
     * `result.index == -1` and `result.value == null` if none were ready.
     */
    @:native("rayzor_select_try_recv")
    public static function tryRecv<T>(channels:Array<Channel<T>>):SelectResult<T>;
}

/**
 * Return value from `Select.recv` / `Select.tryRecv`. Backed by a heap-
 * allocated `SelectResultRaw` struct in the runtime; `index`/`value` are
 * exposed as property accessors that read offsets 0 / 8.
 */
@:native("rayzor::concurrent::SelectResult")
extern class SelectResult<T> {
    /** Index into the original channel array, or -1 when nothing was ready. */
    public var index(get, never):Int;

    /** The value received from `channels[index]`, or null if closed/empty. */
    public var value(get, never):T;

    /** Free the underlying heap-allocated result. Call once after reading. */
    @:native("rayzor_select_result_free")
    public function free():Void;

    @:native("rayzor_select_result_index")
    private function get_index():Int;

    @:native("rayzor_select_result_value")
    private function get_value():T;
}
