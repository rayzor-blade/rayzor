class Node {
    public var left:Node;
    public var key:Int;
    public var height:Int;

    public function new(l:Node, k:Int) {
        left = l;
        key = k;
        if (l != null) {
            height = l.height + 1;
        } else {
            height = 1;
        }
    }
}

class Main {
    static function main() {
        var n1 = new Node(null, 1);
        trace(n1.key);     // 1
        trace(n1.height);  // 1

        var n2 = new Node(n1, 2);
        trace(n2.key);     // 2
        trace(n2.height);  // 2
        trace("done");
    }
}
