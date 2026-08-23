package utest;

/** Minimal stand-in for utest's Assert, covering the members the issue corpus
    actually calls. Counts through unit.ConfCheck so a test using utest scores
    the same way one using unit.Test does.

    utest is a third-party library, not part of the standard library, so the
    real one is not something rayzor is measured against -- but a test that
    calls Assert.equals is still exercising ordinary Haxe, and skipping it
    would hide that coverage. */
class Assert {
    public static function pass(?msg:String):Void {
        unit.ConfCheck.ok();
    }
    public static function fail(?msg:String):Void {
        unit.ConfCheck.fail("Assert.fail" + (msg == null ? "" : ": " + msg));
    }
    public static function isTrue(v:Bool, ?msg:String):Void {
        unit.ConfCheck.ok();
        if (!v) unit.ConfCheck.fail("isTrue");
    }
    public static function isFalse(v:Bool, ?msg:String):Void {
        unit.ConfCheck.ok();
        if (v) unit.ConfCheck.fail("isFalse");
    }
    public static function isNull(v:Dynamic, ?msg:String):Void {
        unit.ConfCheck.ok();
        if (v != null) unit.ConfCheck.fail("isNull");
    }
    public static function notNull(v:Dynamic, ?msg:String):Void {
        unit.ConfCheck.ok();
        if (v == null) unit.ConfCheck.fail("notNull");
    }
    public static function equals(expected:Dynamic, actual:Dynamic, ?msg:String):Void {
        unit.ConfCheck.ok();
        if (expected != actual) unit.ConfCheck.fail("equals: " + expected + " != " + actual);
    }
    public static function notEquals(expected:Dynamic, actual:Dynamic, ?msg:String):Void {
        unit.ConfCheck.ok();
        if (expected == actual) unit.ConfCheck.fail("notEquals: both " + expected);
    }
    public static function floatEquals(expected:Float, actual:Float, ?approx:Float, ?msg:String):Void {
        unit.ConfCheck.ok();
        var d = expected - actual; if (d < 0) d = -d;
        if (d > 1e-9) unit.ConfCheck.fail("floatEquals: " + expected + " != " + actual);
    }
    /** Structural comparison, approximated by rendering both sides. Deep
        equality over arbitrary values needs reflection the corpus does not
        otherwise depend on, and the rendered form separates the cases these
        tests actually distinguish. */
    public static function same(expected:Dynamic, actual:Dynamic, ?recursive:Bool, ?msg:String):Void {
        unit.ConfCheck.ok();
        var a = Std.string(expected);
        var b = Std.string(actual);
        if (a != b) unit.ConfCheck.fail("same: " + a + " != " + b);
    }
    public static function raises(fn:Void->Void, ?type:Dynamic, ?msg:String):Void {
        unit.ConfCheck.ok();
        var threw = false;
        try { fn(); } catch (e:Dynamic) { threw = true; }
        if (!threw) unit.ConfCheck.fail("raises: nothing thrown");
    }
}
