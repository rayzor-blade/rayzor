// Test @:manualDrop — compiler does NOT auto-free, user manages lifetime
// Expected output:
// 42
// Released handle 42
// done

@:manualDrop
class Handle {
    public var fd:Int;
    public function new(fd:Int) { this.fd = fd; }
    public function drop():Void {
        trace("Released handle " + fd);
    }
}

class Main {
    static function main() {
        var h = new Handle(42);
        trace(h.fd);
        h.drop();  // explicit cleanup — user's responsibility
        // No auto-free at scope exit for @:manualDrop
        trace("done");
    }
}
