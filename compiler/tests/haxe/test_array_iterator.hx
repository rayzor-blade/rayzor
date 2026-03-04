class Main {
    static function main() {
        var arr = [10, 20, 30];

        // Test 1: arr.iterator() manual usage
        var it = arr.iterator();
        while (it.hasNext()) {
            trace(it.next());
        }
        // Expected: 10, 20, 30

        // Test 2: for-in over arr.iterator()
        for (x in arr.iterator()) {
            trace(x);
        }
        // Expected: 10, 20, 30

        // Test 3: arr.keyValueIterator() manual usage
        var kvit = arr.keyValueIterator();
        while (kvit.hasNext()) {
            var kv = kvit.next();
            trace(kv.key);
            trace(kv.value);
        }
        // Expected: 0, 10, 1, 20, 2, 30

        // Test 4: Existing for-in still works (regression check)
        for (x in arr) {
            trace(x);
        }
        // Expected: 10, 20, 30

        trace("done");
    }
}
