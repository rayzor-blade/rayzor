package haxe;

class Exception {
    public var message:String;
    public var previous:Exception;
    public var stack(get, set):CallStack;

    /** The call stack text, written where the exception is thrown. */
    @:noCompletion var __nativeStack:String;
    @:noCompletion var __stack:CallStack;

    public function new(message:String, ?previous:Exception, ?native:Any):Void {
        this.message = message;
        this.previous = previous;
    }

    function get_stack():CallStack {
        if (__stack == null)
            __stack = __nativeStack == null ? [] : NativeStackTrace.toHaxe(__nativeStack);
        return __stack;
    }

    function set_stack(stack:CallStack):CallStack {
        return __stack = stack;
    }

    function unwrap():Any {
        return this;
    }

    public function toString():String {
        return message;
    }

    public function details():String {
        if (__nativeStack != null && __nativeStack != "")
            return "Exception: \"" + message + "\"\n" + __nativeStack;
        return message;
    }
}
