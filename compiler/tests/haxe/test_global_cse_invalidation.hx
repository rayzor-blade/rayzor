// Reusing a cached global load is only valid until something writes it.
//
// Local CSE keys a global load by its id alone. That key has to be dropped when
// the global is stored, and when ANY call happens — a callee may store it,
// possibly from a module the pass never sees. Without that, the second read of
// a static returns the value from before the write.
//
// Both shapes below are straight-line: a store between two reads needs no
// module split and no loop, so this is the cheapest possible guard.

class Counter {
    public static var c:Int = 0;

    // Padded past the inline threshold so the store stays inside a callee.
    public static function bump():Void {
        var t = 0;
        for (i in 0...3) t = t + i;
        if (t > 100) Sys.println("unreachable");
        c = c + 1;
    }
}

class Main {
    static var failures = 0;

    static function eqInt(what:String, got:Int, want:Int):Void {
        if (got != want) {
            Sys.println("FAIL " + what + ": got " + got + " want " + want);
            failures++;
        }
    }

    // A store between two reads of the same static.
    static function storeBetween():Int {
        Counter.c = 7;
        var a = Counter.c;
        Counter.c = a + 1;
        var b = Counter.c; // must be 8, not the cached 7
        return a + b;
    }

    // A call between two reads, where the callee does the store.
    static function calleeStore():Int {
        Counter.c = 1;
        var p = Counter.c;
        Counter.bump();
        var q = Counter.c; // must be 2, not the cached 1
        return p + q;
    }

    static function main() {
        eqInt("store between two reads", storeBetween(), 15);
        eqInt("callee store between two reads", calleeStore(), 3);

        // Repeated reads with no write in between SHOULD still collapse — the
        // point is invalidation, not disabling the optimisation.
        Counter.c = 5;
        var x = Counter.c + Counter.c + Counter.c;
        eqInt("reads with no intervening write", x, 15);

        if (failures == 0) {
            Sys.println("PASS: cached global loads are invalidated by stores and calls");
        } else {
            Sys.println("FAILURES: " + failures);
            Sys.exit(1);
        }
    }
}
