@:derive(Debug)
class Point {
    public var x:Int;
    public var y:Int;
    public function new(x:Int, y:Int) { this.x = x; this.y = y; }
}

@:derive(Debug)
class Rect {
    public var topLeft:Point;
    public var bottomRight:Point;
    public var label:String;
    public function new(tl:Point, br:Point, label:String) {
        this.topLeft = tl;
        this.bottomRight = br;
        this.label = label;
    }
}

class Main {
    static function main() {
        var p = new Point(3, 7);
        trace(p.toString());  // Point { x: 3, y: 7 }

        var r = new Rect(new Point(0, 0), new Point(10, 20), "box");
        trace(r.toString());  // Rect { topLeft: Point { x: 0, y: 0 }, bottomRight: Point { x: 10, y: 20 }, label: box }

        trace("done");
    }
}
