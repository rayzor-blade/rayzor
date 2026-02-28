// Test bit shift constant evaluation
package benchmarks;

class TestBitshift {
    // Without type annotation - this should now work!
    static inline var MaxRad = 1 << 16;
    static inline var Typed:Int = 1 << 16;

    public function new() {
        trace("Testing bit shift constant evaluation...");
        trace("MaxRad (untyped) = " + MaxRad);
        trace("Typed:Int = " + Typed);
        trace("Expected: 65536");

        // Test other bitwise operations
        trace("8 >> 2 = " + (8 >> 2)); // 2
        trace("5 & 3 = " + (5 & 3)); // 1
        trace("5 | 3 = " + (5 | 3)); // 7
        trace("5 ^ 3 = " + (5 ^ 3)); // 6
        trace("10 % 3 = " + (10 % 3)); // 1
    }

    public static function main() {
        new TestBitshift();
    }
}
