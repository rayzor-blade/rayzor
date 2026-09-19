// Haxe reads `0xFFFFFFFF` as the Int -1 and a decimal past Int as a Float;
// `9i64` is an Int64. An Int64 static called on a value (`a.toInt()`) takes
// that value as its first parameter, and `toInt` keeps its Haxe body: it
// throws "Overflow" past Int rather than truncating. `method.bind(x)` on an
// instance method calls it on `this`, and a bound Void function returns
// nothing. `a = 1` into an Int64 local a lambda captured widens the Int, and
// `Int64.ofInt(d)` with `d` a Null<Int> unboxes it.
import haxe.Int64;
using haxe.Int64;
class Int64Literals {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function checkD(label:String, got:Dynamic, want:Dynamic) {
        check(label, Std.string(got), Std.string(want));
    }
    function new() {}
    function tryOverflow(a:Int64) { a.toInt(); }
    function exc(label:String, fn:Void->Void) {
        var thrown = false;
        try { fn(); } catch (e:Dynamic) { thrown = true; }
        check(label, Std.string(thrown), "true");
    }
    function run() {
        check("hex wraps", Std.string(0xFFFFFFFF), "-1");
        check("hex min", Std.string(0x80000000), "-2147483648");
        check("decimal past Int is Float", Std.string(4294967295), "4294967295");
        check("Int min literal", Std.string(-2147483648), "-2147483648");
        check("binary wraps", Std.string(0b11111111111111111111111111111111), "-1");
        var big:Int64 = 0x7FFFFFFFFFFFFFFFi64;
        check("i64 suffix", Int64.toStr(big), "9223372036854775807");
        checkD("i64 high", big.high, 0x7FFFFFFF);
        checkD("i64 low", big.low, -1);
        var d:Int64 = 47244640255i64;
        checkD("decimal i64 high", d.high, 10);
        var a:Int64;
        a = Int64.make(10, 5);
        a = 1;
        check("Int into captured Int64", Int64.toStr(a), "1");
        checkD("value receiver toInt", a.toInt(), 1);
        a = Int64.make(0, 0x80000000);
        checkD("word read boxes", a.high, 0);
        checkD("low word boxes", a.low, -2147483648);
        exc("toInt overflows", tryOverflow.bind(a));
        exc("lambda toInt overflows", function() a.toInt());
        a = Int64.make(0xFFFFFFFF, 0x80000000);
        checkD("toInt min", a.toInt(), -2147483648);
        check("parseString", Int64.toStr(Int64.parseString("-23")), "-23");
        check("parseString max", Int64.toStr(Int64.parseString("9223372036854775807")), "9223372036854775807");
        var s = "7";
        var n = s.charCodeAt(0) - '0'.code;
        check("Null<Int> into ofInt", Int64.toStr(Int64.ofInt(n)), "7");
    }
    static function main() {
        new Int64Literals().run();
        Sys.println("CONFORMANCE_OK");
    }
}
