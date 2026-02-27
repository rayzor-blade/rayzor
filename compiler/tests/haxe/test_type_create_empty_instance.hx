class Box {
    public var x:Int;
    public var y:String;

    public function new() {
        x = 7;
        y = "ctor";
    }
}

class Main {
    static function main() {
        var cls = Type.resolveClass("Box");
        var empty:Box = Type.createEmptyInstance(cls);

        // Constructor should NOT run.
        trace(empty.x); // 0
        trace(empty.y == null); // true

        var normal = new Box();
        trace(normal.x); // 7
        trace(normal.y); // ctor
    }
}
