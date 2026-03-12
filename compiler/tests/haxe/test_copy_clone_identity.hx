// Test that Copy and Clone produce independent objects
// and that clone() supports method chaining

@:derive(Copy)
class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) { this.x = x; this.y = y; }
}

@:derive(Clone)
class Named {
    public var name:String;
    public var val:Int;
    public function new(name:String, val:Int) { this.name = name; this.val = val; }
}

class Main {
    static function main() {
        // Copy: let binding independence
        var a = new Point(10, 20);
        var b = a;
        b.x = 99;
        trace(a.x);         // 10
        trace(b.x);         // 99

        // Clone: .clone() independence
        var h = new Named("Hero", 100);
        var c = h.clone();
        c.val = 50;
        trace(h.val);       // 100
        trace(c.val);       // 50
        trace(h.name);      // Hero
        trace(c.name);      // Hero

        trace("done");
    }
}
