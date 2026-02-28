class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) {
        this.x = x;
        this.y = y;
    }
}

class Main {
    static function main() {
        // Test 1: Direct class access (baseline)
        var p = new Point(10, 20);
        trace(p.x);  // 10
        trace(p.y);  // 20

        // Test 2: Dynamic-typed class instance field access
        var d:Dynamic = new Point(3, 7);
        trace(d.x);  // 3
        trace(d.y);  // 7

        trace("done");
    }
}
