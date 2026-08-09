// Dynamic equality semantics.
//
// `a == b` on two Dynamic values compiled to a comparison of the BOX ADDRESSES
// whenever the values arrived as function PARAMETERS: the caller boxes the
// argument, but nothing recorded the parameter symbol as boxed, so the
// value-aware path was skipped. Two separate boxes holding 42 were unequal
// while `a == a` was true, which silently turns any `equals(a, b)` helper into
// "always false".
//
// The value-aware path that did exist unboxed both sides to f64, so it could
// not tell 1 from "1" and read a string as a double. Eq/Ne now go through
// haxe_dynamic_equals, which dispatches on the box tag.

class Main {
    static function eq(a:Dynamic, b:Dynamic):Bool { return a == b; }
    static function ne(a:Dynamic, b:Dynamic):Bool { return a != b; }
    static var bad = 0;
    static function ck(what:String, got:Bool, want:Bool):Void {
        if (got != want) { bad++; Sys.println("FAIL " + what + ": got " + got + " want " + want); }
    }
    static function main() {
        ck("42 == 42", eq(42, 42), true);
        ck("42 == 43", eq(42, 43), false);
        ck("1 == 1.0 (numeric across types)", eq(1, 1.0), true);
        ck("1 == \"1\" (int vs string)", eq(1, "1"), false);
        ck("true == 1 (bool vs int)", eq(true, 1), false);
        ck("\"ab\" == \"ab\"", eq("ab", "ab"), true);
        ck("\"ab\" == \"ac\"", eq("ab", "ac"), false);
        ck("true == true", eq(true, true), true);
        ck("true == false", eq(true, false), false);
        ck("1.5 == 1.5", eq(1.5, 1.5), true);
        ck("null == null", eq(null, null), true);
        ck("null == 0", eq(null, 0), false);
        ck("42 != 42", ne(42, 42), false);
        ck("42 != 43", ne(42, 43), true);
        ck("\"a\" != \"b\"", ne("a", "b"), true);
        // objects: reference identity
        var o1 = {v: 1}; var o2 = {v: 1};
        ck("obj == same obj", eq(o1, o1), true);
        ck("obj == equal-but-distinct obj", eq(o1, o2), false);
        if (bad == 0) Sys.println("PASS: Dynamic equality semantics");
        else { Sys.println("FAILURES: " + bad); Sys.exit(1); }
    }
}
