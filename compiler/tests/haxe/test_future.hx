import rayzor.concurrent.Future;
import rayzor.concurrent.Thread;

class Main {
    static function main() {
        // Test 1: Basic create + await (blocking)
        var f = Future.create(() -> 42);
        trace(f.await()); // 42

        // Test 2: Concurrent futures
        var a = Future.create(() -> 10);
        var b = Future.create(() -> 20);
        trace(a.await() + b.await()); // 30

        // Test 3: isReady before and after await
        var f3 = Future.create(() -> 100);
        var result = f3.await();
        trace(result); // 100

        // Test 4: Multiple sequential awaits
        var x = Future.create(() -> 7 * 6);
        trace(x.await()); // 42

        trace("done");
    }
}
