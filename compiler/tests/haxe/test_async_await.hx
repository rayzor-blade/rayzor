import rayzor.concurrent.Future;

class Main {
    @:async
    static function compute(x:Int):Int {
        return x * 2;
    }

    static function main() {
        // Test 1: Basic @:async + await
        var f = compute(21);
        trace(f.await()); // 42

        // Test 2: Lazy — doesn't execute until await
        var f2 = compute(100);
        trace("not blocked");
        trace(f2.await()); // 200

        // Test 3: Multiple concurrent
        var a = compute(10);
        var b = compute(20);
        trace(a.await() + b.await()); // 60

        trace("done");
    }
}
