class Main {
    static function compute():Int {
        var nowUs:Int = Std.int((Sys.time() * 1e6) % 2000000000.);
        var last:Int = nowUs - 128;
        var gap:Int = nowUs - last;
        var e:Int = 64;
        var delta:Int = gap - e;
        return e + (delta >> 2);
    }

    static function main() {
        var shifted:Int = compute();
        if (shifted != 80) {
            trace("FAIL shift=" + shifted);
            return;
        }

        var signed:Int = -8 >> 1;
        if (signed != -4) {
            trace("FAIL signed=" + signed);
            return;
        }

        trace("PASS cranelift-shift-float-pollution");
    }
}
