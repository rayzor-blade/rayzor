class Main {
    static function main() {
        trace("=== Rayzor Demo Workspace ===");

        // Factorial
        trace('5! = ${MathUtils.factorial(5)}');
        trace('10! = ${MathUtils.factorial(10)}');

        // Fibonacci
        trace('fib(10) = ${MathUtils.fibonacci(10)}');
        trace('fib(20) = ${MathUtils.fibonacci(20)}');

        // Primes
        var primes:Array<Int> = [];
        for (i in 2...50) {
            if (MathUtils.isPrime(i)) {
                primes.push(i);
            }
        }
        trace('Primes under 50: $primes');
    }
}
