// `Null<Int64>` is a box around the native i64: stored boxed, unboxed for a
// word read and for an Int64 local. A lone `'$x'` is a String. A default
// argument may name a preceding parameter or `this`, whether the caller left
// trailing arguments out or skipped an optional one.
import haxe.Int64;
class NullInt64Defaults {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function fSibling(a:Int, b:Int = a):Int return a + b;
    static function fChain(a:Int = 1, b:Int = a, c:Int = b):String return '$a,$b,$c';
    var instanceField = 10;
    function new() {}
    function mThis(x:Int = this.instanceField):Int return x;
    function run() {
        var n:Null<Int64> = null;
        check("null Null<Int64>", Std.string(n == null), "true");
        n = Int64.make(0xf0f0f0f0, 0xefefefef);
        check("boxed high", Std.string(n.high), Std.string(0xf0f0f0f0));
        check("boxed low", Std.string(n.low), Std.string(0xefefefef));
        var m:Int64 = n;
        check("unboxed", Int64.toStr(m), "-1085102592587993105");
        var a:Int64 = 156;
        check("lone interpolation", '$a', "156");
        check("sibling default", Std.string(fSibling(5)), "10");
        check("chain 0", fChain(), "1,1,1");
        check("chain 1", fChain(2), "2,2,2");
        check("chain 2", fChain(2, 3), "2,3,3");
        check("this default", Std.string(mThis()), "10");
        instanceField = 99;
        check("this default reads live", Std.string(mThis()), "99");
    }
    static function main() {
        new NullInt64Defaults().run();
        Sys.println("CONFORMANCE_OK");
    }
}
