// Debug checksum test - trace intermediate values
package benchmarks;

class RGB {
    public var r:Int;
    public var g:Int;
    public var b:Int;

    public function new(inR:Int, inG:Int, inB:Int) {
        r = inR;
        g = inG;
        b = inB;
    }
}

class Complex {
    public var i:Float;
    public var j:Float;

    public function new(inI:Float, inJ:Float) {
        i = inI;
        j = inJ;
    }
}

class DebugChecksum {
    // NOTE: Rayzor requires explicit :Int type for static inline vars
    static inline var SIZE:Int = 25;
    static inline var MaxIterations:Int = 1000;
    static inline var MaxRad:Int = 65536; // 1 << 16
    static inline var width:Int = 875; // 35 * SIZE
    static inline var height:Int = 500; // 20 * SIZE

    public function new() {
        // Test palette creation first
        trace("=== Palette Debug ===");
        trace("Palette[0]: " + paletteDebug(0.0));
        trace("Palette[500]: " + paletteDebug(0.5));
        trace("Palette[1000]: " + paletteDebug(1.0));

        // Test a few specific iterations
        trace("");
        trace("=== Iteration Debug ===");
        var scale = 0.1 / SIZE;

        // Test pixel at (0,0)
        var iteration = computeIteration(0, 0, scale);
        trace("Pixel (0,0) iteration: " + iteration);

        // Test pixel at (100,100)
        iteration = computeIteration(100, 100, scale);
        trace("Pixel (100,100) iteration: " + iteration);

        // Test pixel at (437,250) - middle of image
        iteration = computeIteration(437, 250, scale);
        trace("Pixel (437,250) iteration: " + iteration);

        // Test a small portion (first 10 pixels)
        trace("");
        trace("=== First 10 pixels ===");
        var checksum = 0;
        for (x in 0...10) {
            iteration = computeIteration(x, 0, scale);
            var frac = iteration / MaxIterations;
            var r = Std.int(frac * 255);
            var g = Std.int((1 - frac) * 255);
            var b = Std.int((0.5 - Math.abs(frac - 0.5)) * 2 * 255);
            checksum = checksum + r + g + b;
            trace("x=" + x + " iter=" + iteration + " r=" + r + " g=" + g + " b=" + b + " sum=" + checksum);
        }

        // Full checksum calculation
        trace("");
        trace("=== Full Checksum ===");
        var fullChecksum = 0;
        var palette = new Array<RGB>();
        for (i in 0...MaxIterations + 1)
            palette.push(createPalette(i / MaxIterations));

        for (y in 0...height) {
            for (x in 0...width) {
                iteration = computeIteration(x, y, scale);
                var color = palette[iteration];
                fullChecksum = fullChecksum + color.r + color.g + color.b;
            }
        }
        trace("Full Checksum: " + fullChecksum);
    }

    function computeIteration(x:Int, y:Int, scale:Float):Int {
        var iteration = 0;
        var offset = createComplex(x * scale - 2.5, y * scale - 1);
        var val = createComplex(0.0, 0.0);
        while (complexLength2(val) < MaxRad && iteration < MaxIterations) {
            val = complexAdd(complexSquare(val), offset);
            iteration++;
        }
        return iteration;
    }

    function paletteDebug(frac:Float):String {
        var r = Std.int(frac * 255);
        var g = Std.int((1 - frac) * 255);
        var absVal = Math.abs(frac - 0.5);
        var b = Std.int((0.5 - absVal) * 2 * 255);
        return "r=" + r + " g=" + g + " b=" + b + " (abs=" + absVal + ")";
    }

    public function complexLength2(val:Complex):Float {
        return val.i * val.i + val.j * val.j;
    }

    public inline function complexAdd(val0:Complex, val1:Complex) {
        return createComplex(val0.i + val1.i, val0.j + val1.j);
    }

    public inline function complexSquare(val:Complex) {
        return createComplex(val.i * val.i - val.j * val.j, 2.0 * val.i * val.j);
    }

    public function createComplex(inI:Float, inJ:Float) {
        return new Complex(inI, inJ);
    }

    public function createPalette(inFraction:Float) {
        var r = Std.int(inFraction * 255);
        var g = Std.int((1 - inFraction) * 255);
        var b = Std.int((0.5 - Math.abs(inFraction - 0.5)) * 2 * 255);
        return new RGB(r, g, b);
    }

    public static function main() {
        new DebugChecksum();
    }
}
