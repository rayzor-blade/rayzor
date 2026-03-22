class MathUtils {
    public static function factorial(n:Int):Int {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }

    public static function fibonacci(n:Int):Int {
        if (n <= 1) return n;
        var a = 0;
        var b = 1;
        for (i in 2...n + 1) {
            var c = a + b;
            a = b;
            b = c;
        }
        return b;
    }

    public static function isPrime(n:Int):Bool {
        if (n < 2) return false;
        var i = 2;
        while (i * i <= n) {
            if (n % i == 0) return false;
            i++;
        }
        return true;
    }
}
