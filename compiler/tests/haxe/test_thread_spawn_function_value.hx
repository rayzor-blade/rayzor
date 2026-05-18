// Regression test for compiler fix #1: Function types with Send-only
// captures (or no captures) must be Send themselves.
//
// Before the fix: `TypeKind::Function` was always non-Send, so any
// Thread.spawn capturing a function value — even a bare named function
// reference with no captures — was rejected by Send/Sync validation.
//
// After the fix: Send-ness is tracked per Function type via
// `FunctionEffects.is_send`, set at lambda construction:
// - Bare function refs + Send-only-capture lambdas → is_send = true.
// - Closures capturing non-Send state → is_send = false.
//
// The test compiles + runs successfully — both spawn forms accept the
// function value and the threads join cleanly.

import rayzor.concurrent.Thread;

class Main {
    static function worker():Int {
        return 42;
    }

    static function main() {
        // Bare static function reference captured into the spawn closure.
        // Rejected pre-fix, accepted post-fix.
        var f = worker;
        var t = Thread.spawn(function():Int {
            return f();
        });
        t.join();

        // Sanity check: inline literal with only-primitive captures
        // (always worked; included so the test fails loudly if the new
        // Send rule regresses the primitive path).
        var x = 7;
        var t2 = Thread.spawn(function():Int {
            return x * 6;
        });
        t2.join();
    }
}
