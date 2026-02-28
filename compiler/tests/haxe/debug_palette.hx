// Debug palette indexing
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

class DebugPalette {
    static inline var MaxIterations = 1000;

    public function new() {
        // Create palette
        trace("Creating palette...");
        var palette = new Array<RGB>();
        for (i in 0...10) {
            var frac = i / MaxIterations;
            var r = Std.int(frac * 255);
            var g = Std.int((1 - frac) * 255);
            var b = Std.int((0.5 - Math.abs(frac - 0.5)) * 2 * 255);
            trace("palette[" + i + "]: frac=" + frac + " r=" + r + " g=" + g + " b=" + b);
            palette.push(new RGB(r, g, b));
        }

        trace("");
        trace("Accessing palette...");
        // Access palette at index 4
        var idx = 4;
        trace("idx = " + idx);
        var color = palette[idx];
        if (color != null) {
            trace("palette[4].r = " + color.r);
            trace("palette[4].g = " + color.g);
            trace("palette[4].b = " + color.b);
        } else {
            trace("palette[4] is null!");
        }

        trace("");
        trace("Direct array test...");
        var arr = new Array<Int>();
        for (i in 0...10) {
            arr.push(i * 10);
        }
        trace("arr[0] = " + arr[0]);
        trace("arr[4] = " + arr[4]);
        trace("arr[9] = " + arr[9]);
    }

    public static function main() {
        new DebugPalette();
    }
}
