// Mandelbrot workload from https://github.com/HaxeBenchmarks/benchmark-runner/blob/master/cases/mandelbrot/BMMandelbrotCode.hx
// Used to run official Haxe targets (--interp, HashLink, C++) for comparison.
// The helper is named MandelbrotComplex because HL/C emits Complex as the
// reserved C keyword `_Complex`; the computation is unchanged.

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

class MandelbrotComplex {
    public var i:Float;
    public var j:Float;

    public function new(inI:Float, inJ:Float) {
        i = inI;
        j = inJ;
    }
}

class BMMandelbrotCode {
    static inline var SIZE = 25;
    static inline var MaxIterations = 1000;
    static inline var MaxRad = 1 << 16;
    static inline var width = 35 * SIZE;
    static inline var height = 20 * SIZE;

    public function new() {
        var palette = [];
        for (i in 0...MaxIterations + 1)
            palette.push(createPalette(i / MaxIterations));

        var image = [];
        image[width * height - 1] = null;
        var outPixel = 0;
        var scale = 0.1 / SIZE;
        var checksum = 0;
        for (y in 0...height) {
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

    public function complexLength2(val:MandelbrotComplex):Float {
        return val.i * val.i + val.j * val.j;
    }

    public inline function complexAdd(val0:MandelbrotComplex, val1:MandelbrotComplex) {
        return createComplex(val0.i + val1.i, val0.j + val1.j);
    }

    public inline function complexSquare(val:MandelbrotComplex) {
        return createComplex(val.i * val.i - val.j * val.j, 2.0 * val.i * val.j);
    }

    public function createComplex(inI:Float, inJ:Float) {
        return new MandelbrotComplex(inI, inJ);
    }

    public function createPalette(inFraction:Float) {
        var r = Std.int(inFraction * 255);
        var g = Std.int((1 - inFraction) * 255);
        var b = Std.int((0.5 - Math.abs(inFraction - 0.5)) * 2 * 255);
        return new RGB(r, g, b);
    }

    static function main() {
        new BMMandelbrotCode();
    }
}
