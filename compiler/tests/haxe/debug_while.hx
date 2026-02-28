// Debug while loop
package benchmarks;

class DebugWhile {
    static inline var MaxRad = 1 << 16;
    static inline var MaxIterations = 1000;

    public function new() {
        trace("MaxRad = " + MaxRad);
        trace("MaxIterations = " + MaxIterations);

        var i = 0.0;
        var j = 0.0;
        trace("Initial i=" + i + ", j=" + j);

        var len2 = i * i + j * j;
        trace("Length2 = " + len2);
        trace("Length2 < MaxRad = " + (len2 < MaxRad));

        var iteration = 0;
        trace("iteration < MaxIterations = " + (iteration < MaxIterations));

        // Test combined condition
        trace("Combined condition = " + (len2 < MaxRad && iteration < MaxIterations));

        // Simple while test
        var count = 0;
        while (count < 5) {
            trace("While loop count = " + count);
            count++;
        }
        trace("After while, count = " + count);

        // Test mandelbrot-style while
        trace("");
        trace("=== Mandelbrot while test ===");
        iteration = 0;
        i = 0.0;
        j = 0.0;

        while (i * i + j * j < MaxRad && iteration < MaxIterations) {
            trace("Iteration " + iteration + ": i=" + i + ", j=" + j);
            var new_i = i * i - j * j + (-2.5);
            var new_j = 2.0 * i * j + (-1.0);
            i = new_i;
            j = new_j;
            iteration++;
            if (iteration > 10) break;
        }
        trace("Final iteration = " + iteration);
    }

    public static function main() {
        new DebugWhile();
    }
}
