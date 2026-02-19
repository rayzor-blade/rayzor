class Node {
    public var left:Node;
    public var right:Node;
    public var key:Int;
    public var value:Int;
    public var height:Int;

    public function new(l:Node, k:Int, v:Int, r:Node, h:Int = -1) {
        left = l;
        key = k;
        value = v;
        right = r;
        if (h == -1) {
            var lh = l == null ? 0 : l.height;
            var rh = r == null ? 0 : r.height;
            height = (lh > rh ? lh : rh) + 1;
        } else {
            height = h;
        }
    }
}

class SimpleTree {
    var root:Node;

    public function new() {
        root = null;
    }

    public function set(key:Int, value:Int) {
        root = setLoop(key, value, root);
    }

    function setLoop(k:Int, v:Int, node:Node):Node {
        if (node == null)
            return new Node(null, k, v, null);
        if (node.key == k) {
            return new Node(node.left, k, v, node.right, node.height);
        }
        var nr = setLoop(k, v, node.right);
        return new Node(node.left, node.key, node.value, nr);
    }

    public function get(key:Int):Int {
        var node = root;
        while (node != null) {
            if (node.key == key)
                return node.value;
            node = node.right;
        }
        return 0;
    }
}

class Main {
    static function main() {
        var tree = new SimpleTree();
        tree.set(1, 10);
        tree.set(2, 20);
        tree.set(3, 30);
        trace(tree.get(1));
        trace(tree.get(2));
        trace(tree.get(3));
        trace("done");
    }
}
