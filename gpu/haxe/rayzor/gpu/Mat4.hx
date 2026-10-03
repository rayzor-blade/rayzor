package rayzor.gpu;

/** 4x4 float matrix (column-major). Maps to `mat4x4f` in WGSL. */
class Mat4 {
    /** 16 floats in column-major order: [col0.x, col0.y, col0.z, col0.w, col1.x, ...] */
    public var data:Array<Float>;

    public function new() {
        data = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; // identity
    }

    public static function identity():Mat4 { return new Mat4(); }

    /** Multiply this matrix by a Vec4. */
    public function mulVec4(v:Vec4):Vec4 {
        var d = data;
        return new Vec4(
            d[0]*v.x + d[4]*v.y + d[8]*v.z  + d[12]*v.w,
            d[1]*v.x + d[5]*v.y + d[9]*v.z  + d[13]*v.w,
            d[2]*v.x + d[6]*v.y + d[10]*v.z + d[14]*v.w,
            d[3]*v.x + d[7]*v.y + d[11]*v.z + d[15]*v.w
        );
    }

    /** Multiply two matrices. */
    public function mul(other:Mat4):Mat4 {
        var r = new Mat4();
        var a = data;
        var b = other.data;
        for (col in 0...4) {
            for (row in 0...4) {
                var sum = 0.0;
                for (k in 0...4) {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                r.data[col * 4 + row] = sum;
            }
        }
        return r;
    }

    /** Create a perspective projection matrix. */
    public static function perspective(fov:Float, aspect:Float, near:Float, far:Float):Mat4 {
        var m = new Mat4();
        var f = 1.0 / Math.tan(fov / 2.0);
        var range = near - far;
        m.data = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / range, -1,
            0, 0, 2.0 * far * near / range, 0
        ];
        return m;
    }

    /** Create a translation matrix. */
    public static function translate(x:Float, y:Float, z:Float):Mat4 {
        var m = new Mat4();
        m.data[12] = x;
        m.data[13] = y;
        m.data[14] = z;
        return m;
    }

    /** Create a uniform scale matrix. */
    public static function scale(s:Float):Mat4 {
        var m = new Mat4();
        m.data[0] = s;
        m.data[5] = s;
        m.data[10] = s;
        return m;
    }
}
