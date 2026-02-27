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
        var cls = Type.resolveClass("Point");
        var p:Point = Type.createInstance(cls, [3, 4]);
        trace(p.x); // 3
        trace(p.y); // 4
    }
}
