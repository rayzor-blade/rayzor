// A user method whose name collides with a libc symbol must still compile.
//
// `malloc`, `realloc` and `free` were matched by BARE NAME in the Cranelift
// backend: the declare pass skipped them and a later pass bound the MIR id
// straight to the libc import. A class with a `free()` method therefore had
// its body dropped onto an Import declaration, which cannot be defined, so the
// method got a SIGILL trap stub and silently never ran. In nue that meant
// Q8Cache.free() was dead and the Q8 KV caches were never released.
//
// Only an EXTERN declaration (empty CFG) is the real intrinsic; a bodied
// function merely shares the name.

class Handle {
    public var open:Bool;
    public var freed:Int;
    public function new() { open = true; freed = 0; }
    public function free():Void { open = false; freed++; }
}

// The other two libc names in the same code path.
class Pool {
    public var mallocCalls:Int;
    public var reallocCalls:Int;
    public function new() { mallocCalls = 0; reallocCalls = 0; }
    public function malloc(n:Int):Int { mallocCalls++; return n * 2; }
    public function realloc(n:Int):Int { reallocCalls++; return n + 1; }
}

class Main {
    static var failures = 0;

    // Typed helpers: comparing through Dynamic compares boxes, not values, so
    // an all-equal run still reports every assertion as failed.
    static function eqInt(what:String, got:Int, want:Int):Void {
        if (got != want) {
            failures++;
            Sys.println("FAIL " + what + ": got " + got + " want " + want);
        }
    }

    static function eqBool(what:String, got:Bool, want:Bool):Void {
        if (got != want) {
            failures++;
            Sys.println("FAIL " + what + ": got " + got + " want " + want);
        }
    }

    static function main() {
        var a = new Handle();
        var b = new Handle();
        a.free();
        // The body must actually run, and must not touch the other instance.
        eqBool("a.open", a.open, false);
        eqInt("a.freed", a.freed, 1);
        eqBool("b.open", b.open, true);
        eqInt("b.freed", b.freed, 0);

        // Twice, to catch a stub that happens to no-op once.
        a.free();
        eqInt("a.freed after 2 calls", a.freed, 2);

        var p = new Pool();
        eqInt("malloc returns", p.malloc(21), 42);
        eqInt("realloc returns", p.realloc(41), 42);
        eqInt("malloc counted", p.mallocCalls, 1);
        eqInt("realloc counted", p.reallocCalls, 1);

        if (failures == 0) {
            Sys.println("PASS: libc-named methods compile and run");
        } else {
            Sys.println("FAILURES: " + failures);
            Sys.exit(1);
        }
    }
}
