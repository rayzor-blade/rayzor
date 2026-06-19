// Regression test: an UNTYPED closure literal passed to a function-typed
// parameter must inherit the target signature's parameter types.
//
// Before the fix: `resolve_callee_formal_param_types` only resolved a
// callee via bare scope `lookup_symbol`, which misses a same-class
// static/instance method (`run(...)` == `ThisClass.run(...)`). With no
// expected-param-types hint, the untyped lambda params `(a,b,c)` defaulted
// to `Dynamic` → MIR formals `(*void,*void,*void)`. The lambda body then
// unboxed them via `haxe_unbox_int_ptr`, but the caller passed raw `i32`,
// so unboxing a small int dereferenced address 0x.. → SIGSEGV (exit 139).
// Only EXPLICITLY-typed params, or args at index 0 of an instance method
// resolved via the Field path, escaped it.
//
// After the fix: the Ident branch falls back to the `class_methods` table
// (same as the real call path), so the lambda's untyped params pick up
// `(Int,Int,Int)` and lower to `i32` formals — no unbox, no crash.
//
// Reproduces across Thread.spawn (the WorkerPool.parallelRows shape) since
// that's where the original blocker surfaced, but the inference bug is
// independent of threads.

import rayzor.concurrent.Thread;

class Main {
    // Static method with a function-typed parameter, called UNQUALIFIED
    // from main() — the path that missed the hint pre-fix.
    static function run(fn:(Int, Int, Int) -> Int):Int {
        var threads = new Array<Thread<Int>>();
        for (n in 0...2) {
            var lo = n * 10;
            var hi = n * 10 + 5;
            var node = n;
            threads.push(Thread.spawn(function():Int {
                return fn(lo, hi, node);
            }));
        }
        var sum = 0;
        for (t in threads) sum += t.join();
        return sum;
    }

    // Direct (non-threaded) call of an untyped closure through a typed
    // param — same inference path, no Thread.spawn.
    static function apply2(fn:(Int, Int) -> Int, a:Int, b:Int):Int {
        return fn(a, b);
    }

    static function main() {
        // run() spawns thread 0 -> fn(0,5,0)=5, thread 1 -> fn(10,15,1)=26.
        var s = run((a, b, c) -> a + b + c);
        // apply2 with an untyped 2-arg closure.
        var d = apply2((x, y) -> x * y, 6, 7);

        if (s == 31 && d == 42) {
            Sys.println("PASS untyped-closure-param-inference s=" + s + " d=" + d);
        } else {
            Sys.println("FAIL s=" + s + " (want 31) d=" + d + " (want 42)");
        }
    }
}
