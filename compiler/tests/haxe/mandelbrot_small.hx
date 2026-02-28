// Mandelbrot Benchmark - Small version for testing
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

class MandelbrotSmall {
    // Use smaller size for testing
    static inline var SIZE:Int = 5;
    static inline var MaxIterations:Int = 1000;
    static inline var MaxRad:Int = 65536;
    static inline var width:Int = 175; // 35 * 5
    static inline var height:Int = 100; // 20 * 5

    public function new() {
        trace("Creating palette...");
        var palette = new Array<RGB>();
        for (i in 0...MaxIterations + 1)
            palette.push(createPalette(i / MaxIterations));
        trace("Palette created: " + palette.length + " entries");

        trace("Creating image array...");
        var image = new Array<RGB>();
        image[width * height - 1] = null;
        trace("Image array size: " + image.length);

        var outPixel = 0;
        var scale = 0.1 / SIZE;
        var checksum = 0;
        trace("Starting main loop (width=" + width + ", height=" + height + ")...");

        for (y in 0...height) {
            if (y % 10 == 0) trace("Row " + y + "...");
            for (x in 0...width) {
                var iteration = 0;

                var offset = createComplex(x * scale - 2.5, y * scale - 1);
                var val = createComplex(0.0, 0.0);
                while (complexLength2(val) < MaxRad && iteration < MaxIterations) {
                    val = complexAdd(complexSquare(val), offset);
                    iteration++;
                }

                var color = palette[iteration];
                image[outPixel++] = color;
                checksum = checksum + color.r + color.g + color.b;
            }
        }
        trace("Checksum: " + checksum);
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
        new MandelbrotSmall();
    }
}
