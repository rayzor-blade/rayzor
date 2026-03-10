import rayzor.concurrent.Future;

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

        // Test 5: Non-blocking .then() callback
        Future.create(() -> 99).then((v) -> trace(v)); // 99

        // Test 6: Future.all — spawn in parallel, await all
        var combined = Future.all([
            Future.create(() -> 10),
            Future.create(() -> 20),
            Future.create(() -> 30),
        ]);
        var results = combined.await();
        trace(results[0]); // 10
        trace(results[1]); // 20
        trace(results[2]); // 30

        // Wait for all outstanding futures to complete
        Future.join();
        trace("done");
    }
}
