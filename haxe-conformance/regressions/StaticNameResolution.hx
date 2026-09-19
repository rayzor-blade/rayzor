// A static's name is not its identity. `import haxe.Int64.*` brings the
// owner's statics in as bare names, and a bare `fromFloat(..)` must call
// that owner's -- not another class's same-named static -- and a class's
// own static `fromFloat` is never the runtime row `SIMD4i32.fromFloat`
// shares the name with. Inside a function returning an abstract, a local
// named like one of the abstract's statics (`neg`) stays the local. And a
// same-package type the module references before its file lowers
// (`Int64Helper` inside haxe.Int64) is that type, not `this.Int64Helper`.
import haxe.Int64.*;
class StaticNameResolution {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function fromFloat(f:Float):Int {
        return Std.int(f) * 2;
    }
    static function magnitude(x:Float):haxe.Int64 {
        var neg = x < 0;
        var rest = neg ? -x : x;
        return haxe.Int64.fromFloat(rest);
    }
    static function main() {
        check("wildcard make", toStr(make(0, 5)), "5");
        check("wildcard ofInt", toStr(ofInt(12)), "12");
        check("wildcard add", toStr(add(ofInt(1), ofInt(2))), "3");
        check("own static wins", Std.string(fromFloat(3.0)), "6");
        check("parseString", haxe.Int64.toStr(haxe.Int64.parseString("42")), "42");
        check("Int64.fromFloat", haxe.Int64.toStr(haxe.Int64.fromFloat(3.0)), "3");
        check("negative fromFloat", haxe.Int64.toStr(haxe.Int64.fromFloat(-10.0)), "-10");
        check("local shadows static", toStr(magnitude(-7.0)), "7");
        Sys.println("CONFORMANCE_OK");
    }
}
