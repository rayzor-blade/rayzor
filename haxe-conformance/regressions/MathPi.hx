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
        trace("CONFORMANCE_OK");
    }
}
