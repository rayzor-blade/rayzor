// Two Dynamic operands compare and add by value whatever produced them; an
// anonymous structure's shape is keyed by field names AND types; an untyped
// local waits for its first typed assignment; a caller declared above an
// inferred-return callee sees that type.
private typedef Format = { ?anchorX:Float, ?anchorY:Float };
enum TE { OpStr(s:String); OpIf(e:Void->Dynamic, eif:TE, eelse:Null<TE>); }
class DynamicOperandsAndShapes {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function mk(n:Int):Void->Dynamic return function() { return n; };
    static function field(o:Dynamic, name:String):Dynamic return Reflect.getProperty(o, name);
    static function run(e:TE):String return switch (e) { case OpStr(s): s; case OpIf(_, _, _): "if"; }
    static function main() {
        // forward-declared inferred return, used above its declaration
        check("forward return", cat("x") + " " + (cat("y") == "ay"), "ax true");
        var f = mk(5), g = mk(2);
        var c:Dynamic = f(); var d:Dynamic = g();
        check("dynamic ops", (c > d) + " " + (c < d) + " " + (c == d) + " " + (c + d) + " " + (c - d) + " " + (c * d), "true false false 7 3 10");
        var s:Dynamic = "a"; var t:Dynamic = 1;
        check("dynamic concat", (s + t) + " " + (t + s) + " " + (s + s), "a1 1a aa");
        var o1 = {n: true}; var o2 = {n: 5}; var o3 = {n: "x"};
        check("shapes", Std.string(field(o1, "n")) + " " + Std.string(field(o2, "n")) + " " + Std.string(field(o3, "n")) + " " + (field(o2, "n") > 2), "true 5 x true");
        var eelse;
        if (c > 100) eelse = null; else eelse = OpStr("small");
        check("late-typed local", run(eelse), "small");
        var t1 = new haxe.Template("::if (n > 2)::big::else::small::end::");
        var t2 = new haxe.Template("::if flag::yes::else::no::end::");
        check("template", t1.execute({n: 5}) + t1.execute({n: 1}) + t2.execute({flag: true}) + t2.execute({flag: false}), "bigsmallyesno");
        var obj:Format = haxe.Json.parse("{\"anchorX\": 123.0}");
        check("json typedef", obj.anchorX + "", "123");
        Sys.println("CONFORMANCE_OK");
    }
    static function cat(s) return "a" + s;
}
