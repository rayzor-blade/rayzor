package rayzor.gpu;

/** 3-component float vector. Maps to `vec3f` in WGSL. */
@:gpuStruct
class Vec3 {
    public var x:Float;
    public var y:Float;
    public var z:Float;

    public function new(x:Float, y:Float, z:Float) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public function add(other:Vec3):Vec3 { return new Vec3(x + other.x, y + other.y, z + other.z); }
    public function sub(other:Vec3):Vec3 { return new Vec3(x - other.x, y - other.y, z - other.z); }
    public function scale(s:Float):Vec3 { return new Vec3(x * s, y * s, z * s); }
    public function dot(other:Vec3):Float { return x * other.x + y * other.y + z * other.z; }
    public function cross(other:Vec3):Vec3 {
        return new Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }
    public function length():Float { return Math.sqrt(x * x + y * y + z * z); }
    public function normalize():Vec3 { var l = length(); return new Vec3(x / l, y / l, z / l); }
}
