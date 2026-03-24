package rayzor.gpu;

/** 4-component float vector. Maps to `vec4f` in WGSL. */
@:gpuStruct
class Vec4 {
    public var x:Float;
    public var y:Float;
    public var z:Float;
    public var w:Float;

    public function new(x:Float, y:Float, z:Float, w:Float) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    /** Construct from Vec3 + w component. */
    public static function fromVec3(v:Vec3, w:Float):Vec4 {
        return new Vec4(v.x, v.y, v.z, w);
    }

    public function add(other:Vec4):Vec4 { return new Vec4(x + other.x, y + other.y, z + other.z, w + other.w); }
    public function sub(other:Vec4):Vec4 { return new Vec4(x - other.x, y - other.y, z - other.z, w - other.w); }
    public function scale(s:Float):Vec4 { return new Vec4(x * s, y * s, z * s, w * s); }
    public function dot(other:Vec4):Float { return x * other.x + y * other.y + z * other.z + w * other.w; }
    public function xyz():Vec3 { return new Vec3(x, y, z); }
}
