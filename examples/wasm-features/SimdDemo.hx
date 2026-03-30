import rayzor.SIMD4f;

/**
 * SIMD4f demo — 4-lane f32 vector math.
 * Uses WASM SIMD128 instructions when available.
 *
 * Run: rayzor run --wasm examples/wasm-features/SimdDemo.hx
 */
class SimdDemo {
    static function main() {
        trace("=== SIMD4f Demo ===");

        // Construction
        var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
        var b = SIMD4f.make(5.0, 6.0, 7.0, 8.0);
        trace("a = [" + a[0] + ", " + a[1] + ", " + a[2] + ", " + a[3] + "]");
        trace("b = [" + b[0] + ", " + b[1] + ", " + b[2] + ", " + b[3] + "]");

        // Arithmetic
        var sum = a + b;
        trace("a + b = [" + sum[0] + ", " + sum[1] + ", " + sum[2] + ", " + sum[3] + "]");

        var prod = a * b;
        trace("a * b = [" + prod[0] + ", " + prod[1] + ", " + prod[2] + ", " + prod[3] + "]");

        // Dot product
        var d = a.dot(b);
        trace("a · b = " + d);  // 1*5 + 2*6 + 3*7 + 4*8 = 70

        // Length & normalize
        var len = a.len();
        trace("|a| = " + len);
        var n = a.normalize();
        trace("normalize(a) = [" + n[0] + ", " + n[1] + ", " + n[2] + ", " + n[3] + "]");

        // Cross product (3D)
        var x = SIMD4f.make(1.0, 0.0, 0.0, 0.0);
        var y = SIMD4f.make(0.0, 1.0, 0.0, 0.0);
        var z = x.cross3(y);
        trace("x × y = [" + z[0] + ", " + z[1] + ", " + z[2] + "]");  // [0, 0, 1]

        // Splat & math
        var ones = SIMD4f.splat(4.0);
        var sq = ones.sqrt();
        trace("sqrt([4,4,4,4]) = [" + sq[0] + ", " + sq[1] + ", " + sq[2] + ", " + sq[3] + "]");

        trace("=== Done ===");
    }
}
