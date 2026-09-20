// `...r:T` is `haxe.Rest<T>`: trailing arguments pack into it, a spread
// argument passes an array through, and the Rest reads as an array.
import haxe.Rest;
class RestArgs {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function sum(...r:Int):Int { var s = 0; for (x in r) s += x; return s; }
    static function third(a:Int, b:Int, ...r:Int):Int return r[2];
    static function len(...r:Int):Int return r.length;
    static function join(sep:String, ...parts:String):String return parts.toArray().join(sep);
    static function main() {
        check("pack", Std.string(sum(1, 2, 3)), "6");
        check("empty", Std.string(sum()), "0");
        var a = [4, 5];
        check("spread", Std.string(sum(...a)), "9");
        check("index", Std.string(third(1, 2, 0, 0, 123)), "123");
        check("length", Std.string(len(1, 2, 3, 4)), "4");
        check("strings", join("-", "a", "b", "c"), "a-b-c");
        var local = function(...r:Int):Array<Int> return r.toArray();
        check("closure", local(7, 8).join(","), "7,8");
        Sys.println("CONFORMANCE_OK");
    }
}
