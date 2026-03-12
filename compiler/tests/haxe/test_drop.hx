// Test @:derive(Drop) — user-defined destructor called before Free
// Expected output:
// 1
// drop 1
// 2
// done
// drop 2

@:derive(Drop)
class Connection {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    public function drop():Void {
        trace("drop " + id);
    }
}

class Main {
    static function main() {
        // Test 1: scope exit drop
        var c = new Connection(1);
        trace(c.id);

        // Test 2: reassignment — old value's drop() called before new assigned
        c = new Connection(2);
        trace(c.id);

        // c(2) goes out of scope — drop(2) called at function end
        trace("done");
    }
}
