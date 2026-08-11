// A loop-invariant read is only invariant if the loop cannot write it.
//
// LICM judged instructions by side effects and operand definitions. A read has
// no operands to disqualify it and no side effects, so a global load — or a
// field load — was hoisted into the preheader even when the loop body, or a
// callee, wrote that very location. Every iteration then saw the pre-loop value.

class Obj {
    public var x:Int;
    public function new() { x = 0; }
}

class Counter {
    public static var c:Int = 0;

    // Padded past the inline threshold so the store stays inside the callee.
    public static function bump(o:Obj):Void {
        var t = 0; for (j in 0...3) t = t + j;
        if (t > 100) Sys.println("unreachable");
        Counter.c = Counter.c + 1;
        o.x = o.x + 1;
    }
}

class Main {
    static var failures = 0;

    static function eqInt(what:String, got:Int, want:Int):Void {
        if (got != want) { Sys.println("FAIL " + what + ": got " + got + " want " + want); failures++; }
    }

    // Bottom-tested loop writing a global it also reads.
    static function globalInLoop():Int {
        Counter.c = 0;
        var i = 0; var sum = 0;
        while (true) {
            Counter.c = Counter.c + 1;
            sum = sum + Counter.c;   // 1+2+3+4+5
            i = i + 1;
            if (i >= 5) break;
        }
        return sum;
    }

    // The write happens in a callee, and through a pointer.
    static function fieldViaCallee():Int {
        var o = new Obj();
        var s = 0; var i = 0;
        while (true) {
            Counter.bump(o);
            s = s + o.x;             // 1+2+3+4+5
            i = i + 1;
            if (i >= 5) break;
        }
        return s;
    }

    static function main() {
        eqInt("global read+written in loop", globalInLoop(), 15);
        eqInt("field written by callee in loop", fieldViaCallee(), 15);

        // A genuinely invariant read must STILL hoist — the point is
        // correctness, not disabling the optimisation.
        Counter.c = 3;
        var total = 0;
        for (i in 0...4) total = total + Counter.c;
        eqInt("invariant read still folds", total, 12);

        if (failures == 0) Sys.println("PASS: loop-invariant reads respect what the loop writes");
        else { Sys.println("FAILURES: " + failures); Sys.exit(1); }
    }
}
