class Container<T> {
    public var value:T;
    public function new(v:T) {
        this.value = v;
    }
    public function get():T {
        return this.value;
    }
}

class Pair<A, B> {
    public var first:A;
    public var second:B;
    public function new(a:A, b:B) {
        this.first = a;
        this.second = b;
    }
}

class Main {
    static function main() {
        var intBox = new Container<Int>(42);
        trace(intBox.get());        // 42
        trace(intBox.value);        // 42

        var strBox = new Container<String>("hello");
        trace(strBox.get());        // hello

        var floatBox = new Container<Float>(3.14);
        trace(floatBox.get());      // 3.14

        var pair = new Pair<String, Int>("age", 25);
        trace(pair.first);          // age
        trace(pair.second);         // 25

        trace("done");
    }
}
