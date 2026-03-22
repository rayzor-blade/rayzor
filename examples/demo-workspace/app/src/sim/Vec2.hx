package sim;

class Vec2 {
    public var x:Float;
    public var y:Float;

    public function new(x:Float, y:Float) {
        this.x = x;
        this.y = y;
    }

    public function add(other:Vec2):Vec2 {
        return new Vec2(x + other.x, y + other.y);
    }

    public function scale(s:Float):Vec2 {
        return new Vec2(x * s, y * s);
    }

    public function length():Float {
        return Math.sqrt(x * x + y * y);
    }

    public function distanceTo(other:Vec2):Float {
        var dx = x - other.x;
        var dy = y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    public function toString():String {
        return '(${Math.round(x * 100) / 100}, ${Math.round(y * 100) / 100})';
    }
}
