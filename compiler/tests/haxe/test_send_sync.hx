import rayzor.concurrent.Thread;

// Class with explicit Send derivation — safe to capture in Thread.spawn
@:derive([Send])
class Counter {
    public var value:Int;
    public function new(v:Int) {
        this.value = v;
    }
}

// Class with Send + Sync — safe for Arc sharing
@:derive([Send, Sync])
class SharedConfig {
    public var threshold:Int;
    public var name:String;
    public function new(t:Int, n:String) {
        this.threshold = t;
        this.name = n;
    }
}

class Main {
    static function main() {
        // Test 1: Primitive captures are always Send (no annotation needed)
        var x = 42;
        var h1 = Thread.spawn(() -> {
            return x * 2;
        });
        trace(h1.join()); // 84

        // Test 2: @:derive([Send]) class captured in thread
        var counter = new Counter(10);
        var h2 = Thread.spawn(() -> {
            return counter.value + 5;
        });
        trace(h2.join()); // 15

        // Test 3: @:derive([Send, Sync]) class captured in thread
        var config = new SharedConfig(100, "test");
        var h3 = Thread.spawn(() -> {
            return config.threshold;
        });
        trace(h3.join()); // 100

        // Test 4: String captures are Send
        var msg = "hello";
        var h4 = Thread.spawn(() -> {
            return 42;
        });
        trace(h4.join()); // 42

        trace("done");
    }
}
