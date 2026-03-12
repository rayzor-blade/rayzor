// Test use-after-move detection
// Expected: compiler warning about use after move on line 12 (trace(r.data))
// Expected output (if warnings don't block execution):
// 42

class Resource {
    public var data:Int;
    public function new(d:Int) { data = d; }
}

class Main {
    static function consume(r:Resource):Void { trace(r.data); }
    static function main() {
        var r = new Resource(42);
        consume(r);
        trace(r.data);  // use after move — should warn
    }
}
