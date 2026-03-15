// Fibonacci benchmark — standalone Haxe version for cross-compiler comparison
// Tests: Recursive calls, integer arithmetic, stack depth

class BMFibonacciCode {
    public static function fib(n:Int):Int {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }

    public static function main() {
        var result = fib(40);
        trace("fib(40) = " + result);  // 102334155
    }
}
