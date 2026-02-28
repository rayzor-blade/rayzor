// Test null array assignment
package benchmarks;

class TestNullArray {
    public function new() {
        trace("Test 1: Setting null at index 10");
        var arr = new Array<Int>();
        trace("Array created, length: " + arr.length);
        arr[10] = null;
        trace("After null set, length: " + arr.length);
        trace("arr[10] = " + arr[10]);

        trace("");
        trace("Test 2: Object array with null");
        var arr2:Array<Dynamic> = new Array<Dynamic>();
        trace("Array2 created, length: " + arr2.length);
        arr2[5] = null;
        trace("After null set, length: " + arr2.length);

        trace("All tests passed!");
    }

    public static function main() {
        new TestNullArray();
    }
}
