// Debug static variable division
package benchmarks;

class DebugStaticDiv {
    static var MAX:Int = 1000;
    static inline var INLINE_MAX:Int = 1000;

    public function new() {
        trace("MAX = " + MAX);
        trace("INLINE_MAX = " + INLINE_MAX);

        // Store in local
        var localMax = MAX;
        var localInline = INLINE_MAX;
        trace("localMax = " + localMax);
        trace("localInline = " + localInline);

        // Division with static
        trace("4 / MAX = " + (4 / MAX));
        trace("4 / localMax = " + (4 / localMax));
        trace("4 / INLINE_MAX = " + (4 / INLINE_MAX));
        trace("4 / localInline = " + (4 / localInline));

        // Float division with static
        trace("4.0 / MAX = " + (4.0 / MAX));
        trace("4.0 / localMax = " + (4.0 / localMax));

        // Division by 1000 literal
        trace("4 / 1000 = " + (4 / 1000));

        // Float lhs
        var f:Float = 4.0;
        trace("f / MAX = " + (f / MAX));
        trace("f / localMax = " + (f / localMax));
        trace("f / 1000 = " + (f / 1000));
    }

    public static function main() {
        new DebugStaticDiv();
    }
}
