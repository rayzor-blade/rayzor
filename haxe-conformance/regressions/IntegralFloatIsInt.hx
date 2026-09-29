class IntegralFloatIsInt {
    static function main() {
        var whole:Dynamic = 2.0;
        var fraction:Dynamic = 1.2;
        if (!Std.isOfType(whole, Int)) throw "boxed integral Float";
        if (!Std.isOfType(2.0, Int)) throw "direct integral Float";
        if (Std.isOfType(fraction, Int)) throw "boxed fractional Float";
        if (Std.isOfType(1.2, Int)) throw "direct fractional Float";
        if (Std.isOfType(1e10, Int)) throw "out-of-range Float";
        if (Std.isOfType(Math.NaN, Int)) throw "NaN";
        Sys.println("CONFORMANCE_OK");
    }
}
