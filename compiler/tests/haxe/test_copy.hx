@:derive(Copy)
class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) { this.x = x; this.y = y; }
}

class Main {
    static function modify(p:Point):Void {
        p.x = 999;
    }

    static function main() {
        var a = new Point(3, 7);
        var b = a;              // implicit copy — b is independent
        b.x = 42;
        trace(a.x);            // should be 3, not 42
        trace(b.x);            // should be 42

        modify(a);             // copy at call boundary
        trace(a.x);            // should still be 3

        a = new Point(10, 20); // old a freed, new tracked
        trace(a.x);            // should be 10

        trace("done");
    }
}
