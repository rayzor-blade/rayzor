// Unrolled iterations must not share registers.
//
// The unroller remapped a handful of instruction kinds and cloned everything
// else unchanged, so each copy kept the original's dest register and operands.
// A counted loop touching a static therefore collapsed: every copy defined the
// same register and stored the same stale value.

class Acc {
    public static var total:Int = 0;
    public static var calls:Int = 0;

    public static function add(n:Int):Void {
        var t = 0; for (j in 0...3) t = t + j;
        if (t > 100) Sys.println("unreachable");
        calls = calls + 1;
        total = total + n;
    }
}

class Main {
    static var failures = 0;

    static function eqInt(what:String, got:Int, want:Int):Void {
        if (got != want) { Sys.println("FAIL " + what + ": got " + got + " want " + want); failures++; }
    }

    // Constant trip count with a static accumulator: the shape that collapsed.
    static function staticAccumulator():Int {
        Acc.total = 0;
        var i = 0;
        while (i < 4) { Acc.total = Acc.total + i; i = i + 1; }
        return Acc.total; // 0+1+2+3
    }

    // Same, via a callee, so the body carries a call as well as the globals.
    static function viaCallee():Int {
        Acc.total = 0; Acc.calls = 0;
        var i = 0;
        while (i < 5) { Acc.add(i); i = i + 1; }
        return Acc.total * 100 + Acc.calls; // (0+1+2+3+4)=10, 5 calls
    }

    // A local accumulator alongside an array write, so the body mixes kinds.
    static function localAndArray():Int {
        var a = [0, 0, 0, 0];
        var sum = 0;
        var i = 0;
        while (i < 4) { a[i] = i * 2; sum = sum + a[i]; i = i + 1; }
        return sum; // 0+2+4+6
    }

    static function main() {
        eqInt("static accumulator", staticAccumulator(), 6);
        eqInt("accumulate via callee", viaCallee(), 1005);
        eqInt("local + array in body", localAndArray(), 12);

        if (failures == 0) Sys.println("PASS: unrolled iterations keep distinct registers");
        else { Sys.println("FAILURES: " + failures); Sys.exit(1); }
    }
}
