package unit;

/** Minimal stand-in for the official unit.Test, without the utest dependency.
    Failures print at column 0 so a harness cannot score them as passes. */
class Test {
    public function new() {}

    // `pos` matches upstream's signatures; tests forward their own.
    function eq<T>(v:T, v2:T, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        // Marker first, values second. Rendering a wrong value can crash --
        // a wrong value is often a wrong REPRESENTATION -- and a crash that
        // lands before the marker buries the assertion failure, so the test
        // is scored as a crash rather than the wrong answer it is. Keep both
        // operands at T: handing them to a Dynamic parameter boxes them, and
        // that changes the very comparison being reported on.
        if (v != v2) {
            unit.ConfCheck.fail("eq");
            Sys.println("FAILVALUES " + v + " != " + v2);
        }
    }
    function feq(v:Float, v2:Float, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        var d = v - v2; if (d < 0) d = -d;
        if (d > 1e-9) {
            unit.ConfCheck.fail("feq");
            Sys.println("FAILVALUES " + v + " != " + v2);
        }
    }
    // Upstream is utest's Assert.same, a structural comparison: elements that
    // are distinct objects still match when they print alike.
    function aeq<T>(expected:Array<T>, actual:Array<T>, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        if (expected.length != actual.length) { unit.ConfCheck.fail("aeq length"); return; }
        for (i in 0...expected.length) {
            if (expected[i] != actual[i] && Std.string(expected[i]) != Std.string(actual[i])) {
                unit.ConfCheck.fail("aeq at " + i);
                return;
            }
        }
    }
    function t(v:Bool, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        if (!v) { unit.ConfCheck.fail("t: expected true"); }
    }
    function f(v:Bool, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        if (v) { unit.ConfCheck.fail("f: expected false"); }
    }
    function assert(?message:String, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.fail("assert: " + (message == null ? "" : message));
    }
    function check(v:Bool, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        if (!v) { unit.ConfCheck.fail("check: expected true"); }
    }
    function noAssert(?pos:haxe.PosInfos):Void { unit.ConfCheck.ok(); }
    function unspec(f:Void->Void, ?pos:haxe.PosInfos):Void { unit.ConfCheck.ok(); }
    // Upstream is utest's Assert.contains.
    function allow<T>(v:T, values:Array<T>, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        if (values.indexOf(v) < 0) { unit.ConfCheck.fail("allow: value not in the allowed set"); }
    }
    function hf(c:Class<Dynamic>, n:String, ?pos:haxe.PosInfos):Void {
        t(Type.getInstanceFields(c).indexOf(n) >= 0);
    }
    function nhf(c:Class<Dynamic>, n:String, ?pos:haxe.PosInfos):Void {
        f(Type.getInstanceFields(c).indexOf(n) >= 0);
    }
    function hsf(c:Class<Dynamic>, n:String, ?pos:haxe.PosInfos):Void {
        t(Type.getClassFields(c).indexOf(n) >= 0);
    }
    function nhsf(c:Class<Dynamic>, n:String, ?pos:haxe.PosInfos):Void {
        f(Type.getClassFields(c).indexOf(n) >= 0);
    }
    function exc(fn:Void->Void, ?pos:haxe.PosInfos):Void {
        unit.ConfCheck.ok();
        try { fn(); unit.ConfCheck.fail("exc: no exception"); }
        catch (e:Dynamic) {}
    }
}
