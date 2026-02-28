class Main {
    static function main() {
        // Test 1: Array for-in
        var arr = [10, 20, 30];
        for (x in arr) {
            trace(x);
        }

        // Test 2: Range for-in
        for (i in 0...3) {
            trace(i);
        }

        // Test 3: StringMap for-in (iterates keys)
        var sm = ["hello" => 1, "world" => 2];
        for (key in sm) {
            trace(key);
        }

        // Test 4: IntMap for-in (iterates keys)
        var im = [10 => 100, 20 => 200, 30 => 300];
        for (key in im) {
            trace(key);
        }

        trace("done");
    }
}
