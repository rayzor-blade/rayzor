class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) {
        this.x = x;
        this.y = y;
    }
    public function sum():Int {
        return x + y;
    }
    public function scale(factor:Int):Int {
        return (x + y) * factor;
    }
}

class Main {
    static function main() {
        // Dynamic field access on class
        var d:Dynamic = new Point(3, 7);
        trace(d.x);        // 3
        trace(d.y);        // 7

        // Dynamic method call on class
        trace(d.sum());    // 10
        trace(d.scale(2)); // 20

        // Dynamic arithmetic with class fields
        var total = d.x + d.y;
        trace(total);      // 10

        trace("done");
    }
}
