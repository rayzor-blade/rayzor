import haxe.ds.StringMap;
import haxe.ds.IntMap;

class Main {
    static function main() {
        // Test 1: StringMap via constructor + set
        var m = new StringMap();
        m.set("hello", 42);
        m.set("world", 99);
        trace(m);
        trace(m.get("hello"));
        trace(m.get("world"));
        trace(m.exists("hello"));
        trace(m.exists("missing"));

        // Test 2: IntMap via constructor + set
        var im = new IntMap();
        im.set(1, 100);
        im.set(2, 200);
        trace(im);
        trace(im.get(1));
        trace(im.get(2));
        trace(im.exists(1));
        trace(im.exists(3));

        // Test 3: Map literal syntax (string keys)
        var ml = ["a" => 10, "b" => 20, "c" => 30];
        trace(ml.get("a"));
        trace(ml.get("b"));
        trace(ml.get("c"));

        // Test 4: Map literal syntax (int keys)
        var il = [1 => 111, 2 => 222, 3 => 333];
        trace(il.get(1));
        trace(il.get(2));
        trace(il.get(3));
    }
}
