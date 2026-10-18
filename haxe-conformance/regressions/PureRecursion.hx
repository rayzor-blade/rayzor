// A pure recursive function may compute a repeated call once; the results
// must not change, and a function with side effects must keep every call.
class Main {
    static var calls = 0;

    static function fib(n:Int):Int {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }

    static function trib(n:Int):Int {
        if (n < 3) return n;
        return trib(n - 1) + trib(n - 2) + trib(n - 3);
    }

    static function wraps(n:Int):Int {
        if (n <= 1) return n + 7;
        return wraps(n - 1) * 3 - wraps(n - 2);
    }

    static function twice(n:Int):Int {
        if (n <= 0) return 0;
        return twice(n - 2) + twice(n - 2) + 1;
    }

    static function half(x:Float):Float {
        if (x < 1) return x;
        return half(x - 1) + half(x - 0.5);
    }

    static function counted(n:Int):Int {
        calls += 1;
        if (n <= 1) return n;
        return counted(n - 1) + counted(n - 2);
    }

    static function main() {
        var a = 0, b = 1;
        for (n in 0...41) {
            if (fib(n) != a) throw "fib " + n;
            var t = a + b;
            a = b;
            b = t;
        }
        var t0 = 0, t1 = 1, t2 = 2;
        for (n in 0...26) {
            var expected = n == 0 ? 0 : n == 1 ? 1 : n == 2 ? 2 : 0;
            if (n >= 3) {
                expected = t0 + t1 + t2;
                t0 = t1;
                t1 = t2;
                t2 = expected;
            }
            if (trib(n) != expected) throw "trib " + n;
        }
        var w0 = 7, w1 = 8;
        for (n in 2...30) {
            var w = w1 * 3 - w0;
            w0 = w1;
            w1 = w;
            if (wraps(n) != w) throw "wraps " + n;
        }
        var tw = 0;
        for (k in 1...11) tw = tw * 2 + 1;
        if (twice(20) != tw) throw "twice";
        if (half(4.5) != 17) throw "half " + half(4.5);
        if (counted(20) != 6765 || calls != 21891) throw "counted " + calls;
        Sys.println("CONFORMANCE_OK");
    }
}
