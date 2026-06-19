// Regression test: a Void-returning closure whose body ENDS in a loop must
// not trap.
//
// Before the fix: lambda lowering finalized the implicit return on the
// closure's ENTRY block only (finalize_lambda_terminator_static). When the
// body's last construct was a loop, the entry block already had a terminator
// (the branch into the loop), so finalize returned early — leaving the
// loop-EXIT block with the default `Unreachable` terminator. When the loop
// exited, control fell into `Unreachable`, which executes as `udf` / SIGILL
// (exit 132). Int-returning closures escaped it (the trailing `return`
// terminated the exit block); plain Void FUNCTIONS escaped it
// (ensure_terminator finalizes the current/exit block). Only Void CLOSURES
// ending in a loop trapped — which is exactly the WorkerPool.parallelRows
// worker shape: `(lo,hi,node) -> { var i=lo; while(i<hi){...;i++;} }`.
//
// After the fix: lambda lowering finalizes the block the builder ENDED on
// (the loop-exit / merge block), mirroring how regular functions terminate.
//
// The loop must be the closure's last statement to reproduce the trap, so the
// loop body writes a captured cell each iteration — that both reproduces the
// trap and lets us assert the loop actually ran with correct values.

class Main {
    static function run(fn:(Int, Int) -> Void):Void { fn(0, 8); }

    static function main() {
        var box = [0];

        // while-loop as the closure's final statement (was SIGILL)
        run((lo, hi) -> { var i = lo; while (i < hi) { box[0] = box[0] + i; i++; } });
        var w = box[0]; // 0+1+...+7 = 28

        box[0] = 0;
        // for-loop as the closure's final statement
        run((lo, hi) -> { for (i in lo...hi) box[0] = box[0] + i; });
        var f = box[0]; // 28

        if (w == 28 && f == 28) {
            Sys.println("PASS void-closure-trailing-loop while=" + w + " for=" + f);
        } else {
            Sys.println("FAIL while=" + w + " for=" + f + " (want 28 each)");
        }
    }
}
