package rayzor.concurrent;

/**
 * Thread park/unpark, split out from `Thread` into its own extern so that
 * introducing these members does not perturb the member layout of the
 * heavily cross-referenced `Thread` class (which triggered an order-
 * dependent cross-module id drift). Backed by the same runtime
 * primitives.
 */
@:native("rayzor::concurrent::Parker")
extern class Parker {
    /**
        Register the calling thread for park/unpark and return its park id.
        `unpark(id)` wakes a parked thread; an unpark issued BEFORE the park
        makes the next park return immediately (no lost-wakeup window).
    **/
    @:native("registerParkable")
    public static function registerParkable():Int;

    /** Block until unparked (may also wake spuriously — re-check state). */
    @:native("park")
    public static function park():Void;

    /** Wake the thread registered under `id`. */
    @:native("unpark")
    public static function unpark(id:Int):Void;
}
