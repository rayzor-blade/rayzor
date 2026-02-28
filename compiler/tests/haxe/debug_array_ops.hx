// Debug array operations
package benchmarks;

class DebugArrayOps {
    public function new() {
        trace("Test 1: push()");
        var arr1 = new Array<Int>();
        for (i in 0...5) {
            arr1.push(i);
        }
        trace("arr1.length = " + arr1.length);
        trace("arr1[3] = " + arr1[3]);

        trace("");
        trace("Test 2: Pre-allocate then set");
        var arr2 = new Array<Int>();
        // Pre-allocate by pushing nulls
        for (i in 0...10) {
            arr2.push(0);
        }
        trace("Pre-allocated arr2.length = " + arr2.length);
        arr2[5] = 99;
        trace("arr2[5] = " + arr2[5]);

        trace("");
        trace("Test 3: Direct index on empty array");
        var arr3 = new Array<Int>();
        trace("arr3.length before = " + arr3.length);
        // This might crash
        arr3[0] = 1;
        trace("arr3[0] = " + arr3[0]);

        trace("All tests passed!");
    }

    public static function main() {
        new DebugArrayOps();
    }
}
