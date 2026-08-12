// A variable reassigned inside a loop is live across the back edge, even when
// its last textual use sits before the reassignment. Statement-index last-use
// analysis cannot see that, and dropping there frees the object the next
// iteration reads.

class P {
    public var v:Int;

    public function new(n:Int) {
        v = n;
    }
}

class Main {
    static function main() {
        var keep = new P(7);
        var cur = keep;
        var sum = 0;
        var i = 0;
        while (i < 4) {
            // Last textual use of `cur` in the body — but the back edge makes
            // the object assigned below live into the next iteration.
            sum += cur.v;
            cur = new P(i);
            i++;
        }

        var ok = true;
        if (sum != 10) {
            trace("expected sum 10, got " + sum);
            ok = false;
        }
        if (keep.v != 7) {
            trace("expected keep.v 7, got " + keep.v);
            ok = false;
        }
        if (!ok) {
            Sys.exit(1);
        }
        trace("loop reassigned alias ok");
    }
}
