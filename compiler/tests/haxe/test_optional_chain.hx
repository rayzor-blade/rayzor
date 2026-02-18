class Node {
    public var value:Int;
    public var next:Node;

    public function new(v:Int) {
        value = v;
        next = null;
    }

    public function getValue():Int {
        return value;
    }
}

class Main {
    static function main() {
        // Basic optional field access on non-null
        var n = new Node(42);
        trace(n?.value);            // 42

        // Optional field access on null - returns 0 (null as Int)
        var n2:Node = null;
        trace(n2?.value);           // 0

        // Optional method call on non-null
        trace(n?.getValue());       // 42

        // Optional method call on null
        trace(n2?.getValue());      // 0

        // Chained optional access
        n.next = new Node(99);
        trace(n?.next?.value);      // 99

        // Chained optional where middle is null
        var n3 = new Node(7);
        trace(n3?.next?.value);     // 0

        // Optional on non-null reference field
        trace(n?.next?.getValue()); // 99

        trace("done");
    }
}
