package unit;

/** Minimal stand-in for the official unit.Test, without the utest dependency.
    Failures print at column 0 so a harness cannot score them as passes. */
class Test {
    public function new() {}

    function eq<T>(v:T, v2:T):Void {
        ConfCheck.ok();
        if (v != v2) { ConfCheck.fail("eq: " + v + " != " + v2); }
    }
    function feq(v:Float, v2:Float):Void {
        ConfCheck.ok();
        var d = v - v2; if (d < 0) d = -d;
        if (d > 1e-9) { ConfCheck.fail("feq: " + v + " != " + v2); }
    }
    function aeq<T>(expected:Array<T>, actual:Array<T>):Void {
        ConfCheck.ok();
        if (expected.length != actual.length) { ConfCheck.fail("aeq length"); return; }
        for (i in 0...expected.length) {
            if (expected[i] != actual[i]) { ConfCheck.fail("aeq at " + i); return; }
        }
    }
    function t(v:Bool):Void {
        ConfCheck.ok();
        if (!v) { ConfCheck.fail("t: expected true"); }
    }
    function f(v:Bool):Void {
        ConfCheck.ok();
        if (v) { ConfCheck.fail("f: expected false"); }
    }
    function assert(?message:String):Void {
        ConfCheck.fail("assert: " + (message == null ? "" : message));
    }
    function check(v:Bool):Void {
        ConfCheck.ok();
        if (!v) { ConfCheck.fail("check: expected true"); }
    }
    function noAssert():Void { ConfCheck.ok(); }
    function unspec(f:Void->Void):Void { ConfCheck.ok(); }
    function exc(fn:Void->Void):Void {
        ConfCheck.ok();
        try { fn(); ConfCheck.fail("exc: no exception"); }
        catch (e:Dynamic) {}
    }
}
