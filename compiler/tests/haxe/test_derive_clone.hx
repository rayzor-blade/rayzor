@:derive(Clone)
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
        var p2 = p.clone();
        trace(p2.x);
        trace(p2.y);

        // Mutation independence
        p2.x = 99;
        trace(p.x);
        trace(p2.x);
    }
}
