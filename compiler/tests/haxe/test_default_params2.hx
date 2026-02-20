class Node {
    public var left:Node;
    public var right:Node;
    public var key:Int;
    public var height:Int;

    public function new(l:Node, k:Int, r:Node, h:Int = -1) {
        left = l;
        key = k;
        right = r;
        if (h == -1) {
            var lh = 0;
            var rh = 0;
            if (left != null) lh = left.height;
            if (right != null) rh = right.height;
            height = (lh > rh ? lh : rh) + 1;
        } else {
            height = h;
        }
    }
}

class Main {
    static function main() {
        var n1 = new Node(null, 1, null);     // h defaults to -1
        trace(n1.key);     // 1
        trace(n1.height);  // 1

        var n2 = new Node(null, 2, null, 5);  // h = 5
        trace(n2.key);     // 2
        trace(n2.height);  // 5

        var n3 = new Node(n1, 3, n2);         // h defaults to -1, computed from children
        trace(n3.key);     // 3
        trace(n3.height);  // 6 (max(1,5) + 1)
        trace("done");
    }
}
