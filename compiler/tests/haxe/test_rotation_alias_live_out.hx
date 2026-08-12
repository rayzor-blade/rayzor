// A loop-carried allocation is released at the latch, but only when no other
// name for the object outlives the loop. Here `keep` aliases the object the
// first iteration would release, and reads it afterwards.

class P {
    public var x:Float;
    public var y:Float;

    public function new(a:Float, b:Float) {
        x = a;
        y = b;
    }
}

class Main {
    static function main() {
        var v = new P(0.0, 0.0);
        var keep = v;
        var i = 0;
        var acc = 0.0;
        while (i < 5) {
            v = new P(v.x + 1.0, v.y + 2.0);
            acc += v.x;
            i++;
        }

        var ok = true;
        if (acc != 15.0) {
            trace("expected acc 15, got " + acc);
            ok = false;
        }
        // Reads the object the rotation would have released on iteration one.
        if (keep.x != 0.0 || keep.y != 0.0) {
            trace("alias read after loop saw " + keep.x + "," + keep.y);
            ok = false;
        }
        if (!ok) {
            Sys.exit(1);
        }
        trace("rotation alias live-out ok");
    }
}
