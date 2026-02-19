class Node {
    public var next:Node;
    public var value:Int;

    public function new(v:Int) {
        value = v;
        next = null;
    }
}

class Main {
    static function main() {
        var n1 = new Node(1);
        var n2 = new Node(2);
        n1.next = n2;
        trace(n1.value);       // 1
        trace(n1.next.value);  // 2
        trace("done");
    }
}
