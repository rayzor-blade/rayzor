// Std.string, string concat and Sys.println agree on what a value prints as,
// whatever holds it: a Dynamic box from Reflect.field reads through its tag
// (it was taken for a raw handle and its payload pointer read as a field),
// an enum prints its constructor by RTTI (it printed its address), a class
// instance held as Dynamic reaches its own toString (it printed "void"), and
// Sys.println of a non-String stringifies first (it passed the raw value).
private enum Color { Red; Green(n:Int); }
private class WithTs { public function new() {} public function toString() return "WithTs!"; }
private class Plain { public var v = 5; public var s = "sv"; public var n:Null<Int> = null; public function new() {} }
private class Holder { public var kid = new Plain(); public var xs = [1, 2]; public function new() {} }
class DynamicStringify {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function main() {
        var h = new Holder();
        var k:Dynamic = Reflect.field(h, "kid");
        check("box field int", "" + k.v, "5");
        check("box field string", "" + k.s, "sv");
        check("box nested", "" + Std.string(k.v), "5");
        var d:Dynamic = h;
        check("dynamic chain", "" + d.kid.v, "5");
        check("null field", Std.string(Reflect.field(h.kid, "n") == null), "true");
        check("typeof array", Std.string(Type.typeof(Reflect.field(h, "xs"))), "TClass(Array)");
        check("enum concat", "" + Red, "Red");
        check("enum std", Std.string(Green(3)), "Green(3)");
        var w:Dynamic = new WithTs();
        check("toString via dynamic", Std.string(w), "WithTs!");
        check("toString via concat", "" + w, "WithTs!");
        check("plain class", Std.string(Std.string(new Plain()) != "void"), "true");
        var sb = new StringBuf(); sb.add("ab"); sb.add(1);
        var dsb:Dynamic = sb;
        check("stdlib toString via dynamic", Std.string(dsb), "ab1");
        var de:Dynamic = new haxe.Exception("boom");
        check("exception toString via dynamic", Std.string(de), "boom");
        var i:Dynamic = 7;
        check("int box", "" + i, "7");
        Sys.println(1.5);
        Sys.println(Green(2));
        Sys.println(Type.typeof(w));
        Sys.println("CONFORMANCE_OK");
    }
}
