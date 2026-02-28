// Debug division
package benchmarks;

class DebugDivision {
    static inline var MaxIterations = 1000;

    public function new() {
        var i:Int = 4;
        trace("i = " + i);
        trace("MaxIterations = " + MaxIterations);

        // Direct division
        var result1 = i / MaxIterations;
        trace("i / MaxIterations = " + result1);

        // Cast to float first
        var result2:Float = i / MaxIterations;
        trace("(Float) i / MaxIterations = " + result2);

        // Explicit float cast
        var result3 = (1.0 * i) / MaxIterations;
        trace("(1.0 * i) / MaxIterations = " + result3);

        // Both floats
        var fi:Float = i;
        var fm:Float = MaxIterations;
        var result4 = fi / fm;
        trace("fi / fm = " + result4);

        // Division by literal
        var result5 = i / 1000;
        trace("i / 1000 = " + result5);

        var result6 = i / 1000.0;
        trace("i / 1000.0 = " + result6);

        // Float / int
        var f:Float = 4.0;
        var result7 = f / 1000;
        trace("4.0 / 1000 = " + result7);
    }

    public static function main() {
        new DebugDivision();
    }
}
