// Debug untyped vs typed static var
package benchmarks;

class DebugUntypedStatic {
    static inline var UNTYPED = 1000;
    static inline var TYPED:Int = 1000;

    public function new() {
        trace("UNTYPED = " + UNTYPED);
        trace("TYPED = " + TYPED);

        trace("4 / UNTYPED = " + (4 / UNTYPED));
        trace("4 / TYPED = " + (4 / TYPED));

        var i = 4;
        trace("i / UNTYPED = " + (i / UNTYPED));
        trace("i / TYPED = " + (i / TYPED));
    }

    public static function main() {
        new DebugUntypedStatic();
    }
}
