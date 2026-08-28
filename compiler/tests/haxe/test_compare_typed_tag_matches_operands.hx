// `haxe_reflect_compare_typed` reads BOTH of its i64 slots the way one type
// tag says to. Whenever the tag is picked from one operand's static type and
// the other operand has a different representation, the runtime reads a box
// address as a value, or a box as a HaxeString. Both answer wrongly and the
// program exits 0, so these are checked, not traced.

class Main {
    static var failures = 0;

    static function check(label:String, ok:Bool):Void {
        if (!ok) {
            failures++;
            Sys.println("FAIL " + label);
        }
    }

    // The shape every generic test helper takes: both operands are T.
    static function eq<T>(a:T, b:T):Bool {
        return a == b;
    }

    static function cmp<T>(a:T, b:T):Int {
        return Reflect.compare(a, b);
    }

    static function main():Void {
        // T = Dynamic: two separate boxes holding the same value.
        var d1:Dynamic = 5;
        var d2:Dynamic = 5;
        check("generic eq on two Dynamic ints", eq(d1, d2));
        var s:Dynamic = "hel" + "lo";
        var s2:Dynamic = "hello";
        check("generic eq on two Dynamic strings", eq(s, s2));
        var other:Dynamic = 6;
        check("generic ne on two Dynamic ints", !eq(d1, other));

        // T = Null<Int>: the box is a DynamicValue, not a HaxeString.
        var n1:Null<Int> = 7;
        var n2:Null<Int> = 7;
        var n3:Null<Int> = 8;
        check("generic eq on two Null<Int>", eq(n1, n2));
        check("generic ne on two Null<Int>", !eq(n1, n3));

        // Reflect.compare with operands of different representations: the tag
        // must not be read off the first argument alone.
        var di:Dynamic = 5;
        check("compare Int against Dynamic Int", Reflect.compare(5, di) == 0);
        check("compare Dynamic Int against Int", Reflect.compare(di, 5) == 0);
        var ds:Dynamic = "hello";
        check("compare String against Dynamic String", Reflect.compare("hello", ds) == 0);

        // Controls: the cases that already worked must keep working.
        check("compare two Ints", Reflect.compare(5, 5) == 0);
        check("compare Floats orders", Reflect.compare(0.1, 0.2) < 0);
        check("compare Strings orders", Reflect.compare("a", "b") < 0);
        check("generic compare Floats orders", cmp(0.2, 0.1) > 0);
        check("generic eq on Ints", eq(5, 5));
        check("generic ne on Ints", !eq(5, 6));
        check("generic eq on Strings", eq("hel" + "lo", "hello"));
        check("generic eq on Floats", eq(1.5, 1.5));
        check("generic ne on Floats", !eq(1.5, 1.7));
        check("generic eq on Bools", eq(true, true));

        if (failures == 0) {
            Sys.println("PASS compare_typed tag matches operands");
        } else {
            Sys.println("FAILURES: " + failures);
            Sys.exit(1);
        }
    }
}
