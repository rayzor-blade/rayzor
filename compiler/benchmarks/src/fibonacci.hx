// Fibonacci Benchmark
// Recursive implementation — tests function call overhead and tier promotion
//
// Tests: Recursive calls, integer arithmetic, stack depth

package benchmarks;

class Fibonacci {
    public static function fib(n:Int):Int {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }

    public static function main() {
        var result = fib(40);
        trace("fib(40) = " + result);  // 102334155
    }
}
