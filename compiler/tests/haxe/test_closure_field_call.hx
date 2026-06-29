// Regression: calling a closure stored in an object field directly via
// `obj.fieldFn(args)` must invoke the closure (indirect call), not dispatch
// a (nonexistent) method named after the field, which traps at runtime.
class Entry {
    public var name:String;
    public var body:Void->Int;
    public function new(n:String, b:Void->Int) { name = n; body = b; }
}
class Main {
    static function check(cond:Bool, msg:String):Void {
        if (!cond) { trace("FAIL: " + msg); Sys.exit(1); }
    }
    static function main() {
        var e = new Entry("a", () -> 42);
        check(e.body() == 42, "direct field-closure call e.body()");

        // Same, iterated over an array (the test-runner shape).
        var arr = [new Entry("x", () -> 1), new Entry("y", () -> 2), new Entry("z", () -> 3)];
        var sum = 0;
        for (t in arr) sum += t.body();
        check(sum == 6, "field-closure call in for-in loop");

        trace("test_closure_field_call OK");
    }
}
