class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) { this.x = x; this.y = y; }
}

class Main {
    static function print2d(p:{x:Int, y:Int}) {
        trace(p.x);
        trace(p.y);
    }
    static function main() {
        // Scenario 2: anon wider → narrower (deferred, index remap)
        var wide = {a: 99, x: 10, y: 20, z: 30};
        var narrow:{x:Int, y:Int} = wide;
        trace(narrow.x);
        trace(narrow.y);

        // Scenario 3: class → anon (deferred, GEP access)
        var pt = new Point(3, 7);
        var p2:{x:Int, y:Int} = pt;
        trace(p2.x);
        trace(p2.y);

        // Scenario 5: function param (materialization at call boundary)
        print2d(pt);

        // Function param with wider anon
        print2d({x: 100, y: 200, z: 300});

        // Scenario: reference semantics (class-backed, mutation visible)
        pt.x = 42;
        trace(p2.x);

        trace("done");
    }
}
