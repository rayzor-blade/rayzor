// Debug array set
package benchmarks;

class DebugArraySet {
    public function new() {
        trace("=== Array Set Debug ===");

        // Create and fill array
        var arr = new Array<Int>();
        arr.push(10);
        arr.push(20);
        arr.push(30);
        arr.push(40);
        arr.push(50);

        trace("Initial array:");
        trace("arr[0] = " + arr[0]);
        trace("arr[1] = " + arr[1]);
        trace("arr[2] = " + arr[2]);
        trace("arr[3] = " + arr[3]);
        trace("arr[4] = " + arr[4]);

        trace("");
        trace("Setting arr[2] = 999...");
        arr[2] = 999;

        trace("After set:");
        trace("arr[0] = " + arr[0]);
        trace("arr[1] = " + arr[1]);
        trace("arr[2] = " + arr[2]);
        trace("arr[3] = " + arr[3]);
        trace("arr[4] = " + arr[4]);

        trace("");
        trace("Direct comparison: arr[2] == 999 -> " + (arr[2] == 999));
        trace("Direct comparison: arr[2] == 30 -> " + (arr[2] == 30));
    }

    public static function main() {
        new DebugArraySet();
    }
}
