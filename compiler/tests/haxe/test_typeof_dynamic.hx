// Regression: Type.typeof(v) where v is statically Dynamic must lower to a real
// value (not a "Return with no value" trap-stub). Verifies primitive cases are
// classified correctly. (Object/class values in a Dynamic still classify as
// TUnknown pending box-type-id consistency — a separate, deeper issue.)
class Main {
    static function tof(v:Dynamic):String { return Std.string(Type.typeof(v)); }
    static function check(got:String, want:String):Void {
        if (got != want) { trace("FAIL: got " + got + " want " + want); Sys.exit(1); }
    }
    static function main() {
        check(tof(5), "TInt");
        check(tof(3.5), "TFloat");
        check(tof(true), "TBool");
        // Also exercise the inline (non-return) Dynamic form.
        var d:Dynamic = 42;
        check(Std.string(Type.typeof(d)), "TInt");
        trace("test_typeof_dynamic OK");
    }
}
