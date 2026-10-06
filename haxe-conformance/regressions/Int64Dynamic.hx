import haxe.Int64;

class Int64Dynamic {
    static function check(label:String, got:Dynamic, want:Dynamic) {
        if (got != want) throw label + ": " + got + " != " + want;
    }

    static function boxed(value:Int64):Dynamic {
        return value;
    }

    static function roundTrip(value:Dynamic, high:Int, low:Int, text:String) {
        check("Int64 identity", Int64.isInt64(value), true);
        check("not Int", Std.isOfType(value, Int), false);
        check("class token", Std.isOfType(value, Type.getClass(value)), true);
        var native:Int64 = value;
        check("high", native.high, high);
        check("low", native.low, low);
        check("Dynamic high", value.high, high);
        check("Dynamic low", value.low, low);
        check("Dynamic string", Std.string(value), text);
    }

    static function main() {
        var value:Dynamic = Int64.make(1, 1);
        roundTrip(value, 1, 1, "4294967297");
        roundTrip(boxed(Int64.make(-1, -1)), -1, -1, "-1");
        var values:Array<Dynamic> = [Int64.make(0x7fffffff, -1), Int64.make(0x80000000, 0)];
        roundTrip(values[0], 0x7fffffff, -1, "9223372036854775807");
        roundTrip(values[1], 0x80000000, 0, "-9223372036854775808");
        check("object", Int64.isInt64({}), false);
        check("Int", Int64.isInt64(1), false);
        check("null", Int64.isInt64(null), false);
        Sys.println("CONFORMANCE_OK");
    }
}
