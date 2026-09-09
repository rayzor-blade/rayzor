class FloatNegation {
    static inline function negate(v:Float):Float { return -v; }
    static function main() {
        var value:Float = 36;
        if (negate(value) != -36.0) throw "promoted integer";
        if (negate(1.5) != -1.5) throw "float";
        if (negate(-2) != 2.0) throw "negative integer";
        trace("CONFORMANCE_OK");
    }
}
