// A do-while runs its body once before the first test, and its body is
// inside the loop: continue goes to the condition, break to the exit.
class DoWhileLoops {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function firstReturn():Int {
        do { return 7; } while (true);
    }
    static function main() {
        var k = 0; var i = 0;
        do { i++; if (i & 1 == 0) continue; k++; } while (i++ < 10);
        check("continue", i + " " + k, "12 6");

        var n = 0;
        do { n++; if (n == 3) break; } while (true);
        check("break", "" + n, "3");

        var once = 0;
        do once++ while (false);
        check("runs once", "" + once, "1");

        var x:Int;
        do { x = 5; } while (false);
        check("assigned in body", "" + x, "5");

        var s = 0; var j = 0;
        do { j++; if (j % 2 == 0) continue; s += j; } while (j < 6);
        check("continue carries values", s + " " + j, "9 6");
        check("return in body", "" + firstReturn(), "7");

        var fs = [];
        var m = 0;
        do { var c = m; fs.push(() -> c); m++; } while (m < 3);
        check("fresh local per pass", [for (f in fs) f()].join(","), "0,1,2");
        trace("CONFORMANCE_OK");
    }
}
