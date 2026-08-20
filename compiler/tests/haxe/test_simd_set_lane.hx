// `v.set(lane, x)` wrote lane 0 whatever `lane` was, on SIMD4f and SIMD4i32
// alike — VectorInsert needs a constant index and the lowering hard-coded one.
// `get` had already been fixed the same way; this is its mirror.
import rayzor.SIMD4f;
import rayzor.SIMD4i32;
class test_simd_set_lane {
    static function main():Void {
        for (lane in 0...4) {
            var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
            var b = a.set(lane, 99.0);
            Sys.println("f32 set(" + lane + ") -> [" + b.get(0) + "," + b.get(1) + "," + b.get(2) + "," + b.get(3) + "]");
        }
        for (lane in 0...4) {
            var a = SIMD4i32.make(1, 2, 3, 4);
            var b = a.set(lane, 99);
            Sys.println("i32 set(" + lane + ") -> [" + b.get(0) + "," + b.get(1) + "," + b.get(2) + "," + b.get(3) + "]");
        }
    }
}
