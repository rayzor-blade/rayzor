class Main {
    static function main() {
        // Basic Null<Int> assignment
        var x:Null<Int> = 42;
        trace(x);                 // 42

        var y:Null<Int> = null;
        trace(y);                 // null

        // Null coalescing with Null<T>
        var z = x ?? 0;
        trace(z);                 // 42

        var w = y ?? -1;
        trace(w);                 // -1

        trace("done");
    }
}
