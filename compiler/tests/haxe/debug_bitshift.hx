// Debug bit shift
package benchmarks;

class DebugBitshift {
    static inline var SHIFT_16 = 1 << 16;
    static inline var SHIFT_8 = 1 << 8;
    static inline var CONST_65536 = 65536;
    static var SHIFT_16_VAR = 1 << 16;

    public function new() {
        trace("static inline var SHIFT_16 = 1 << 16: " + SHIFT_16);
        trace("static inline var SHIFT_8 = 1 << 8: " + SHIFT_8);
        trace("static inline var CONST_65536 = 65536: " + CONST_65536);
        trace("static var SHIFT_16_VAR = 1 << 16: " + SHIFT_16_VAR);

        var local = 1 << 16;
        trace("var local = 1 << 16: " + local);

        trace("Direct 1 << 16: " + (1 << 16));
        trace("Direct 1 << 8: " + (1 << 8));
    }

    public static function main() {
        new DebugBitshift();
    }
}
