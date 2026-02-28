class Main {
    static function maybeNull():String {
        return null;
    }

    static function main() {
        // Test null coalescing ?? with non-nullable primitives (short-circuits to LHS)
        var y:Int = 10;
        var z = y ?? 99;
        trace(z);  // 10

        // Test ?? with string
        var s1 = "hello" ?? "default";
        trace(s1);  // hello

        trace("done");
    }
}
