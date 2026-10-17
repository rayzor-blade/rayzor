// A self-call returned directly, or added to a value computed before it, may
// become a loop; the values and the order of the calls must not change.
class Main {
    static var order:Array<Int> = [];

    static function fib(n:Int):Int {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }

    static function sumTo(n:Int):Int {
        if (n == 0) return 0;
        order.push(n);
        return n + sumTo(n - 1);
    }

    static function countdown(n:Int, acc:Int):Int {
        if (n == 0) return acc;
        return countdown(n - 1, acc * 3 + n);
    }

    static function wraps(n:Int):Int {
        if (n == 0) return 0x7ffffff0;
        return wraps(n - 1) + 0x10000000;
    }

    static function fibSlow(n:Int):Int {
        var a = 0, b = 1;
        for (i in 0...n) {
            var t = a + b;
            a = b;
            b = t;
        }
        return a;
    }

    static function main() {
        for (n in 0...25)
            if (fib(n) != fibSlow(n)) throw "fib " + n;
        if (sumTo(100) != 5050) throw "sumTo";
        if (order.length != 100 || order[0] != 100 || order[99] != 1) throw "call order";
        var expected = 0;
        for (n in 1...8) expected = expected * 3 + (8 - n);
        if (countdown(7, 0) != expected) throw "tail call";
        var wrapped:Int = 0x7ffffff0;
        for (i in 0...20) wrapped = wrapped + 0x10000000;
        if (wraps(20) != wrapped) throw "wrapping accumulation";
        Sys.println("CONFORMANCE_OK");
    }
}
