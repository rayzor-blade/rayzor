/**
 * Math + String + Array demo — core operations on WASM.
 *
 * Run: rayzor run --wasm examples/wasm-features/MathDemo.hx
 */
class MathDemo {
    static function main() {
        trace("=== Math & String Demo ===");

        // Math operations (WASM runtime uses libm)
        trace("sqrt(144) = " + Math.sqrt(144));
        trace("floor(3.7) = " + Math.floor(3.7));
        trace("ceil(3.2) = " + Math.ceil(3.2));
        trace("abs(-42) = " + Math.abs(-42));

        // String operations
        var greeting = "Hello" + ", " + "WASM" + "!";
        trace(greeting);
        trace("Length: " + greeting.length);

        // Array operations
        var arr = [10, 20, 30, 40, 50];
        trace("Array: length=" + arr.length);
        trace("arr[2] = " + arr[2]);
        arr.push(60);
        trace("After push: length=" + arr.length);

        // Fibonacci
        var a = 0;
        var b = 1;
        var i = 0;
        while (i < 20) {
            var fib = a + b;
            a = b;
            b = fib;
            i++;
        }
        trace("Fibonacci(20) = " + b);

        trace("=== Done ===");
    }
}
