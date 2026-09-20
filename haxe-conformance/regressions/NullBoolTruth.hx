// A `?q:Bool` parameter is Null<Bool>: read as a condition, negated or
// combined it is its value, and an omitted argument reads as false.
class NullBoolTruth {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function cond(?q:Bool):String return if (q) "T" else "F";
    static function tern(?q:Bool):String return q ? "T" : "F";
    static function not(?q:Bool):Bool return !q;
    static function both(?q:Bool, r:Bool):Bool return q && r;
    static function either(?q:Bool, r:Bool):Bool return q || r;
    static function guard(?q:Bool):String return switch (1) { case 1 if (q): "T"; case _: "F"; }
    static function loop(?q:Bool):Int { var n = 0; while (q && n < 3) n++; return n; }
    static function main() {
        check("if", cond(true) + cond(false) + cond(), "TFF");
        check("ternary", tern(true) + tern(false) + tern(), "TFF");
        check("not", Std.string(not(true)) + Std.string(not(false)) + Std.string(not()), "falsetruetrue");
        check("and", Std.string(both(true, true)) + Std.string(both(false, true)) + Std.string(both(null, true)), "truefalsefalse");
        check("or", Std.string(either(true, false)) + Std.string(either(false, false)) + Std.string(either(null, true)), "truefalsetrue");
        check("guard", guard(true) + guard(false) + guard(), "TFF");
        check("while", Std.string(loop(true)) + Std.string(loop()), "30");
        check("htmlEscape", StringTools.htmlEscape("a\"b<", false) + StringTools.htmlEscape("a\"b", true), "a\"b&lt;a&quot;b");
        Sys.println("CONFORMANCE_OK");
    }
}
