// Regression test for compiler fix #3: numerical primitives are Copy,
// so passing them to a function or mutating in a loop must NOT trigger
// the use-after-move warning.
//
// Pre-fix: every Variable reference in a function-call argument got
// `add_move` recorded by populate_ownership_expr, regardless of type.
// `i++` in a `while (i < items)` loop was the typical false positive.
//
// After: the diagnostic-emission loop in check_ownership_violations
// skips violations whose variable type is Copy (Int, Float, Bool, or
// any class deriving Copy). Move tracking on actual heap-allocated
// classes still surfaces real bugs.

class Main {
    static function take(n:Int):Int { return n + 1; }

    static function main() {
        // Int mutated in a while loop — pre-fix produced the warning.
        var i = 0;
        var sum = 0;
        while (i < 5) {
            sum += i;
            i++;
        }
        trace(sum); // 10

        // Float used after being passed to a function.
        var f:Float = 3.5;
        trace(take(Std.int(f)));
        trace(f);   // 3.5

        // Bool used after pass-through.
        var b = true;
        if (b) trace("ok");
        trace(b);   // true
    }
}
