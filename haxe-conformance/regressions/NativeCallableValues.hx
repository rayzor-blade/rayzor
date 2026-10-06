class NativeCallableValues {
    static function apply(f:Float->Float, value:Float):Float return f(value);
    static function main() {
        var cos = Math.cos;
        if (cos(0.0) != 1.0) throw "cos variable";
        var object = {sqrt: Math.sqrt};
        if (object.sqrt(9.0) != 3.0) throw "sqrt field";
        if (apply(Math.abs, -2.5) != 2.5) throw "abs argument";
        var round = Math.round;
        if (round(2.8) != 3) throw "integer return";
        var isNaN = Math.isNaN;
        if (!isNaN(Math.NaN) || isNaN(1.0)) throw "boolean return";
        var stringify = Std.string;
        if (stringify(7) != "7") throw "Dynamic argument";
        var compare = Reflect.compare;
        if (compare(2, 3) >= 0) throw "Dynamic comparator";
        Sys.println("CONFORMANCE_OK");
    }
}
