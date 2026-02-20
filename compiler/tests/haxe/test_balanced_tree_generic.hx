// Generic balanced tree test — mimics haxe.ds.BalancedTree structure

class TreeNode<K, V> {
    public var left:TreeNode<K, V>;
    public var right:TreeNode<K, V>;
    public var key:K;
    public var value:V;
    public var height:Int;

    public function new(l:TreeNode<K, V>, k:K, v:V, r:TreeNode<K, V>, h:Int = -1) {
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

class BalancedTree<K, V> {
    var root:TreeNode<K, V>;

    public function new() {
        root = null;
    }

    public function set(key:K, value:V) {
        root = setLoop(key, value, root);
    }

    function setLoop(k:K, v:V, node:TreeNode<K, V>):TreeNode<K, V> {
        if (node == null)
            return new TreeNode<K, V>(null, k, v, null);
        var c = Reflect.compare(k, node.key);
        if (c == 0) {
            return new TreeNode<K, V>(node.left, k, v, node.right, node.height);
        }
        if (c < 0) {
            var nl = setLoop(k, v, node.left);
            return new TreeNode<K, V>(nl, node.key, node.value, node.right);
        }
        var nr = setLoop(k, v, node.right);
        return new TreeNode<K, V>(node.left, node.key, node.value, nr);
    }

    public function get(key:K):V {
        var node = root;
        while (node != null) {
            var c = Reflect.compare(key, node.key);
            if (c == 0)
                return node.value;
            if (c < 0)
                node = node.left;
            else
                node = node.right;
        }
        return null;
    }

    public function exists(key:K):Bool {
        var node = root;
        while (node != null) {
            var c = Reflect.compare(key, node.key);
            if (c == 0)
                return true;
            if (c < 0)
                node = node.left;
            else
                node = node.right;
        }
        return false;
    }
}

class Main {
    static function main() {
        // Test with Int keys, Int values
        var intTree = new BalancedTree<Int, Int>();
        intTree.set(3, 30);
        intTree.set(1, 10);
        intTree.set(2, 20);
        trace(intTree.get(1));
        trace(intTree.get(2));
        trace(intTree.get(3));
        trace(intTree.exists(2));
        trace(intTree.exists(5));

        // Test with String keys, Int values
        var strTree = new BalancedTree<String, Int>();
        strTree.set("banana", 2);
        strTree.set("apple", 1);
        strTree.set("cherry", 3);
        trace(strTree.get("apple"));
        trace(strTree.get("banana"));
        trace(strTree.get("cherry"));
        trace(strTree.exists("banana"));
        trace(strTree.exists("mango"));

        trace("done");
    }
}
