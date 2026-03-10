@:derive(Debug)
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
        var p = new Point(3, 7);
        trace(p.toString());
    }
}
