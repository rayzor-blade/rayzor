package haxe;

import haxe.NativeStackTrace;

class Exception {
    public var message:String;
    public var previous:Exception;

    public function new(message:String, ?previous:Exception, ?native:Any):Void {
        this.message = message;
        this.previous = previous;
    }

    @:keep
    public inline function toString():String {
        return message;
    }

    public inline function details():String {
        var stack = NativeStackTrace.exceptionStack();
        if (stack != null && stack != "")
            return "Exception: \"" + message + "\"\n" + stack;
        return message;
    }
}
