class Container<T> {
    public var value:T;
    public function new(v:T) {
        this.value = v;
    }
}

class Main {
    static function main() {
        var intBox = new Container<Int>(42);
        trace(intBox.value);
        trace("done");
    }
}
