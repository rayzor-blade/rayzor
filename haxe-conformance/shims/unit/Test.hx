package unit;

/** Minimal stand-in for the official unit.Test, without the utest dependency.
    Failures print at column 0 so a harness cannot score them as passes. */
class Test {
    public function new() {}

    function eq<T>(v:T, v2:T):Void {
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
    function feq(v:Float, v2:Float):Void {
        unit.ConfCheck.ok();
        var d = v - v2; if (d < 0) d = -d;
        if (d > 1e-9) {
            unit.ConfCheck.fail("feq");
            Sys.println("FAILVALUES " + v + " != " + v2);
        }
    }
    function aeq<T>(expected:Array<T>, actual:Array<T>):Void {
        unit.ConfCheck.ok();
        if (expected.length != actual.length) { unit.ConfCheck.fail("aeq length"); return; }
        for (i in 0...expected.length) {
            if (expected[i] != actual[i]) { unit.ConfCheck.fail("aeq at " + i); return; }
        }
    }
    function t(v:Bool):Void {
        unit.ConfCheck.ok();
        if (!v) { unit.ConfCheck.fail("t: expected true"); }
    }
    function f(v:Bool):Void {
        unit.ConfCheck.ok();
        if (v) { unit.ConfCheck.fail("f: expected false"); }
    }
    function assert(?message:String):Void {
        unit.ConfCheck.fail("assert: " + (message == null ? "" : message));
    }
    function check(v:Bool):Void {
        unit.ConfCheck.ok();
        if (!v) { unit.ConfCheck.fail("check: expected true"); }
    }
    function noAssert():Void { unit.ConfCheck.ok(); }
    function unspec(f:Void->Void):Void { unit.ConfCheck.ok(); }
    function exc(fn:Void->Void):Void {
        unit.ConfCheck.ok();
        try { fn(); unit.ConfCheck.fail("exc: no exception"); }
        catch (e:Dynamic) {}
    }
}
