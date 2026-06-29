// Regression: a variable modified in a catch handler must not leak its value
// to the continuation when the try body completes normally (no throw). The
// continuation needs a merge phi over the try-exit / catch-exit paths.
class Main {
    static function check(cond:Bool, msg:String):Void {
        if (!cond) { trace("FAIL: " + msg); Sys.exit(1); }
    }
    static function main() {
        // No throw: the catch must NOT run, so `fail` stays 0.
        var pass = 0; var fail = 0;
        try { pass += 1; } catch (e:Dynamic) { fail += 1; }
        check(pass == 1, "pass==1");
        check(fail == 0, "catch counter leaked on no-throw path");

        // Throw mid-try: increment-after-throwing-work pattern (test runner shape).
        var p2 = 0; var f2 = 0;
        for (i in 0...3) {
            try {
                if (i == 1) throw "boom";
                p2 += 1;
            } catch (e:Dynamic) {
                f2 += 1;
            }
        }
        check(p2 == 2, "p2==2 (passes counted)");
        check(f2 == 1, "f2==1 (failure counted)");

        trace("test_trycatch_merge_phi OK");
    }
}
