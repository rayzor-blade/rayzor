class ForwardClassMembers {
    static var pending = new LaterTask(() -> pending.stop());
    static function check(value:Bool, label:String):Void { if (!value) throw label; }
    static function main():Void {
        var task:LaterTask = cast pending;
        task.run();
        check(task.stopped, "forward inferred static receiver in callback");
        var box = new LaterBox<Float>(3.25);
        check(box.value == 3.25, "forward generic field read");
        box.value = 4.75;
        check(box.value == 4.75, "forward generic field write");
        check(box.amount == 2.5, "forward property initializer");
        box.amount = 5.5;
        check(box.amount == 5.5, "forward property setter");

        var received = 0.0;
        var signal = new LaterSignal<Float>(value -> received = value);
        signal.callback(1.75);
        check(received == 1.75, "forward inherited generic callback");
        var callback = signal.callback;
        callback(2.25);
        check(received == 2.25, "capturing callback read as value");

        var integer = 0;
        var ints = new LaterSignal<Int>(value -> integer = value);
        ints.callback(37);
        check(integer == 37, "forward integer callback");
        var text = "";
        var strings:LaterSignal<String> = new LaterSignal(value -> text = value);
        strings.callback("received");
        check(text == "received", "contextual constructor callback");
        Sys.println("CONFORMANCE_OK");
    }
}

class LaterBox<T> {
    public var value:T;
    public var amount(default, set):Float = 2.5;
    public function new(value:T) this.value = value;
    function set_amount(value:Float):Float return amount = value;
}
class LaterSignal<T> extends LaterCallbackStore<T->Void> {
    public function new(handler:T->Void) this.callback = value -> handler(value);
}
class LaterCallbackStore<F> {
    public var callback:F;
}

class LaterTask {
    public var stopped:Bool = false;
    var callback:Void->Void;
    public function new(callback:Void->Void) this.callback = callback;
    public function run():Void callback();
    public function stop():Void stopped = true;
}
