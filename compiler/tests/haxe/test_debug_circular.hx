// Test @:derive(Debug) with self-referential types
// Expected output:
// Node { value: 1, next: Node { value: 2, next: <object> } }

@:derive(Debug)
class Node {
    public var value:Int;
    public var next:Node;
    public function new(v:Int) { value = v; next = null; }
}

class Main {
    static function main() {
        var a = new Node(1);
        var b = new Node(2);
        a.next = b;
        b.next = a;  // circular
        trace(a.toString());
    }
}
