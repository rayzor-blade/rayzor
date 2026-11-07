class Circle {
    public static var PI:Float = 3;
    public static function tau():Float return PI * 2;
}

class MathPi {
    static function area(radius:Float):Float return Math.PI * radius * radius;

    static function main() {
        if (Math.abs(Math.PI - 3.141592653589793) > 1e-15) throw "Math.PI value";
        if (Math.abs(area(2) - 12.566370614359172) > 1e-12) throw "Math.PI in a function";
        // A user's own PI is not Math.PI.
        var PI = 4.0;
        if (PI != 4.0) throw "local PI";
        if (Circle.tau() != 6) throw "static PI";
        if (Math.cos(Math.PI) != -1) throw "cos of Math.PI";
        if (("pi " + Math.PI).indexOf("3.14159") != 3) throw "Math.PI concatenated";
        if (Std.int(Math.PI * 2) != 6) throw "Std.int of Math.PI";
        var turns = 2.5 / (2 * Math.PI);
        if (Math.abs(turns - 0.3978873577297384) > 1e-15) throw "Math.PI in a division";
        var pi = Math.PI;
        var scale = function(n:Int) return n * pi;
        if (Math.abs(scale(3) - 9.42477796076938) > 1e-12) throw "Math.PI captured by a closure";
        trace("CONFORMANCE_OK");
    }
}
