class GenericTree<K, V> {
    var root_key:K;
    var root_value:V;
    var has_root:Bool;

    public function new() {
        has_root = false;
    }

    public function set(key:K, value:V) {
        root_key = key;
        root_value = value;
        has_root = true;
    }

    public function get(key:K):V {
        if (has_root) {
            var c = Reflect.compare(key, root_key);
            trace(c);  // should be 0 for matching key
            if (c == 0)
                return root_value;
        }
        return null;
    }
}

class Main {
    static function main() {
        // Direct Reflect.compare test
        trace(Reflect.compare("a", "a"));  // 0
        trace(Reflect.compare("a", "b"));  // -1

        var tree = new GenericTree<String, String>();
        tree.set("hello", "world");
        var result = tree.get("hello");
        trace(result);  // world
        trace("done");
    }
}
