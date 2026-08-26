@:publicFields
private abstract NestedStaticAbstract(Int) from Int {
    static function make(f:Float):Int {
        return Std.int(f);
    }
}

class Main {
    static function main() {
        var got = NestedStaticAbstract.make(1.2);
        if (got != 1) {
            trace("FAIL: expected 1, got " + got);
            Sys.exit(1);
        }
        trace("PASS");
    }
}
