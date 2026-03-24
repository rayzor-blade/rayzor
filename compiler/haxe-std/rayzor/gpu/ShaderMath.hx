package rayzor.gpu;

/**
 * Shader math functions — work on CPU and transpile to WGSL builtins.
 *
 * In @:shader classes, these are transpiled to their WGSL equivalents:
 * `ShaderMath.mix(a, b, t)` → `mix(a, b, t)` in WGSL.
 */
class ShaderMath {
    /** Linear interpolation: a*(1-t) + b*t */
    public static inline function mix(a:Float, b:Float, t:Float):Float {
        return a * (1.0 - t) + b * t;
    }

    /** Clamp value to [min, max] range. */
    public static inline function clamp(x:Float, lo:Float, hi:Float):Float {
        return Math.max(lo, Math.min(hi, x));
    }

    /** Hermite smoothstep interpolation. */
    public static function smoothstep(edge0:Float, edge1:Float, x:Float):Float {
        var t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }

    /** Step function: 0.0 if x < edge, 1.0 otherwise. */
    public static inline function step(edge:Float, x:Float):Float {
        return if (x < edge) 0.0 else 1.0;
    }

    /** Fractional part: x - floor(x). */
    public static inline function fract(x:Float):Float {
        return x - Math.floor(x);
    }

    /** Sign: -1, 0, or 1. */
    public static inline function sign(x:Float):Float {
        return if (x > 0) 1.0 else if (x < 0) -1.0 else 0.0;
    }

    /** Dot product for Vec3. */
    public static inline function dot3(a:Vec3, b:Vec3):Float {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    /** Cross product for Vec3. */
    public static inline function cross(a:Vec3, b:Vec3):Vec3 {
        return a.cross(b);
    }

    /** Normalize a Vec3. */
    public static inline function normalize3(v:Vec3):Vec3 {
        return v.normalize();
    }

    /** Length of a Vec3. */
    public static inline function length3(v:Vec3):Float {
        return v.length();
    }

    /** Reflect vector v around normal n. */
    public static function reflect(v:Vec3, n:Vec3):Vec3 {
        var d = 2.0 * dot3(v, n);
        return v.sub(n.scale(d));
    }
}
