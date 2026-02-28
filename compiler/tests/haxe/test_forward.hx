@:forward(push, length, pop)
abstract SafeArray(Array<Int>) {
}

@:forward
abstract AllForward(Array<Int>) {
}

class Main {
    static function main() {
        var plain = new Array();
        var arr:SafeArray = cast plain;

        // Test push and length via @:forward(push, length, pop)
        arr.push(10);
        arr.push(20);
        arr.push(30);
        trace(arr.length);    // 3

        // Test pop
        var last = arr.pop();
        trace(last);          // 30
        trace(arr.length);    // 2

        // Test that underlying array is shared
        trace(plain.length);  // 2

        // Test @:forward (forward all)
        var plain2 = new Array();
        var all:AllForward = cast plain2;
        all.push(100);
        all.push(200);
        trace(all.length);    // 2
        trace(all.pop());     // 200
        trace(all.length);    // 1
    }
}
