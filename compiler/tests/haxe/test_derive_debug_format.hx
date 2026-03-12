@:derive(Debug)
@:debugFormat("({x}, {y})")
class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) { this.x = x; this.y = y; }
}

@:derive(Debug)
@:debugFormat("[{name} #{id}]")
class Entity {
    public var name:String;
    public var id:Int;
    public var active:Bool;
    public function new(name:String, id:Int, active:Bool) {
        this.name = name;
        this.id = id;
        this.active = active;
    }
}

class Main {
    static function main() {
        var p = new Point(3, 7);
        trace(p.toString());  // (3, 7)

        var e = new Entity("Player", 42, true);
        trace(e.toString());  // [Player #42]

        trace("done");
    }
}
