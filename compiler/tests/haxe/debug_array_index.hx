// Debug array indexing
package benchmarks;

class DebugArrayIndex {
    public function new() {
        trace("Testing array indexing...");

        var arr = new Array<Int>();
        trace("Created empty array, length: " + arr.length);

        // Set element at index 10 (should auto-expand)
        arr[10] = 42;
        trace("Set arr[10] = 42, length: " + arr.length);
        trace("arr[10] = " + arr[10]);

        // Test larger index
        trace("Setting arr[100]...");
        arr[100] = 99;
        trace("Set arr[100] = 99, length: " + arr.length);

        // Test even larger
        trace("Setting arr[1000]...");
        arr[1000] = 123;
        trace("Set arr[1000] = 123, length: " + arr.length);

        // Test very large
        trace("Setting arr[17500]...");
        arr[17500] = 456;
        trace("Set arr[17500] = 456, length: " + arr.length);

        trace("Test passed!");
    }

    public static function main() {
        new DebugArrayIndex();
    }
}
