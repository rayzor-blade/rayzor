// Behavioural test: user wrapper class with @:autoDeref enables the
// `wrapper.field` / `wrapper.method()` deref-coercion machinery on
// any user class that exposes a `get()` method. Previously hardcoded
// to rayzor.concurrent.Arc / MutexGuard.

class Counter {
    public var value: Int;
    public function new(v: Int) { this.value = v; }
    public function bumped(): Int { return this.value + 1; }
}

@:autoDeref
class CounterBox {
    var inner: Counter;
    public function new(v: Counter) { this.inner = v; }
    public function get(): Counter { return this.inner; }
}

class Main {
    static function main() {
        var box = new CounterBox(new Counter(41));
        trace(box.value);     // 41   (field deref: box.value → box.get().value)
        trace(box.bumped());  // 42   (method deref)
    }
}
