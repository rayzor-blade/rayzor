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
        var p1 = new Point(1, 2);
        var p2 = new Point(3, 4);
        var p3 = new Point(5, 6);

        var map = new haxe.ds.ObjectMap<Point, Int>();

        // set + get
        map.set(p1, 10);
        map.set(p2, 20);
        map.set(p3, 30);
        trace(map.get(p1)); // 10
        trace(map.get(p2)); // 20

        // exists — pointer identity, not structural equality
        trace(map.exists(p1)); // true
        trace(map.exists(new Point(1, 2))); // false

        // remove
        trace(map.remove(p3)); // true
        trace(map.exists(p3)); // false

        // for-in key=>value iteration
        var sum = 0;
        for (key => value in map) {
            sum += value;
        }
        trace(sum); // 30

        trace("done");
    }
}
