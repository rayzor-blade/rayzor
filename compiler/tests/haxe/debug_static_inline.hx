// Debug static inline var
package benchmarks;

class DebugStaticInline {
    static inline var INLINE_1000 = 1000;
    static var STATIC_1000 = 1000;
    static inline var INLINE_EXPR = 35 * 25;

    public function new() {
        trace("INLINE_1000 = " + INLINE_1000);
        trace("STATIC_1000 = " + STATIC_1000);
        trace("INLINE_EXPR (35*25) = " + INLINE_EXPR);

        // Check what type these are
        var a = INLINE_1000;
        var b = STATIC_1000;
        trace("var a = INLINE_1000: " + a);
        trace("var b = STATIC_1000: " + b);

        // Division tests
        trace("4 / INLINE_1000 = " + (4 / INLINE_1000));
        trace("4 / STATIC_1000 = " + (4 / STATIC_1000));

        // Check if null
        if (INLINE_1000 == null) {
            trace("INLINE_1000 is null!");
        }
        if (INLINE_1000 == 0) {
            trace("INLINE_1000 is 0!");
        }

        // Type check
        var x:Dynamic = INLINE_1000;
        trace("Dynamic INLINE_1000 = " + x);
    }

    public static function main() {
        new DebugStaticInline();
    }
}
