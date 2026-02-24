package haxe;

class Exception {
    public var message:String;
    public var previous:Exception;

    public function new(message:String, ?previous:Exception, ?native:Any):Void {
        this.message = message;
        this.previous = previous;
    }

    public function toString():String {
        return message;
    }

    public function details():String {
        return message;
    }
}
