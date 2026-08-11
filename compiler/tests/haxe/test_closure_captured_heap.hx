// A heap value captured by a closure must stay alive for the closure's life.
//
// The MIR free-insertion pass decides ownership by walking the uses of an
// allocation. It had no arm for closure creation, so a captured array looked
// dead at the end of the frame that made it and was freed there — the returned
// closure then read and wrote freed memory.

class Main {
    static var failures = 0;

    static function eqInt(what:String, got:Int, want:Int):Void {
        if (got != want) { Sys.println("FAIL " + what + ": got " + got + " want " + want); failures++; }
    }

    // The capture outlives the frame that allocated it.
    static function makeCounter(start:Int):Int->Int {
        var buf = [start];
        return function(k:Int) { buf[0] = buf[0] + k; return buf[0]; };
    }

    // Two closures over separate buffers must not alias.
    static function main() {
        var a = makeCounter(10);
        var b = makeCounter(100);

        eqInt("first call", a(1), 11);
        eqInt("state persists", a(2), 13);
        eqInt("independent capture", b(5), 105);
        eqInt("still independent", a(0), 13);

        // Many live closures at once: each keeps its own buffer.
        var fs = new Array<Int->Int>();
        for (i in 0...8) fs.push(makeCounter(i * 10));
        var sum = 0;
        for (i in 0...8) sum = sum + fs[i](1);
        eqInt("eight independent captures", sum, 288); // (0+10+..+70) + 8

        if (failures == 0) Sys.println("PASS: closure-captured heap values outlive their frame");
        else { Sys.println("FAILURES: " + failures); Sys.exit(1); }
    }
}
