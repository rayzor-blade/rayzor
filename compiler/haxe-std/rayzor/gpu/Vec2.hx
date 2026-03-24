package rayzor.gpu;

/** 2-component float vector. Maps to `vec2f` in WGSL. */
@:gpuStruct
class Vec2 {
    public var x:Float;
    public var y:Float;

    public function new(x:Float, y:Float) {
        this.x = x;
        this.y = y;
    }

    public function add(other:Vec2):Vec2 { return new Vec2(x + other.x, y + other.y); }
    public function sub(other:Vec2):Vec2 { return new Vec2(x - other.x, y - other.y); }
    public function scale(s:Float):Vec2 { return new Vec2(x * s, y * s); }
    public function dot(other:Vec2):Float { return x * other.x + y * other.y; }
    public function length():Float { return Math.sqrt(x * x + y * y); }
    public function normalize():Vec2 { var l = length(); return new Vec2(x / l, y / l); }
}
