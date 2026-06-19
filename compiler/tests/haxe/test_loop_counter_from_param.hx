// Regression test: a loop counter initialized from a parameter (or any
// value that aliases another SSA register) must still advance.
//
// Before the fix: loop-phi construction in hir_to_mir excluded function
// parameters from getting a header phi by comparing REGISTERS
// (`symbol_map.get(sym) == Some(&p.reg)`). A local initialized directly from
// a parameter — `var i = lo` — aliases `lo`'s register instead of emitting a
// copy, so the counter was misclassified as the parameter and got no phi.
// The increment was computed then discarded and the condition read the
// parameter's entry value forever: band-iteration loops
// (`var i = lo; while (i < hi)` and `for (i in lo...hi)`) spun infinitely.
// This blocked every pure-Haxe band-parallel kernel (WorkerPool.parallelRows
// gives each worker a `[lo, hi)` slice).
//
// After the fix: parameters are identified by SYMBOL KIND, so an aliasing
// local is not excluded and receives a proper loop phi. Fixed across while /
// do-while / for-loop lowering.

class Main {
    // var i = lo; while (i < hi)  — counter init from a param
    static function sumWhile(lo:Int, hi:Int):Int {
        var s = 0;
        var i = lo;
        while (i < hi) { s += i; i++; }
        return s;
    }

    // for (i in lo...hi)  — desugars to a param-initialized counter
    static function sumFor(lo:Int, hi:Int):Int {
        var s = 0;
        for (i in lo...hi) s += i;
        return s;
    }

    static function main() {
        // sum 2..10 = 2+3+4+5+6+7+8+9 = 44
        var w = sumWhile(2, 10);
        var f = sumFor(2, 10);

        // Same, but inside a closure invoked through a function-typed param
        // (the WorkerPool worker shape: a captured closure with a band loop).
        var apply = function(fn:(Int, Int) -> Int, a:Int, b:Int):Int return fn(a, b);
        var c = apply((lo, hi) -> { var s = 0; var i = lo; while (i < hi) { s += i; i++; } return s; }, 2, 10);

        if (w == 44 && f == 44 && c == 44) {
            Sys.println("PASS loop-counter-from-param while=" + w + " for=" + f + " closure=" + c);
        } else {
            Sys.println("FAIL while=" + w + " for=" + f + " closure=" + c + " (want 44 each)");
        }
    }
}
