// A tree built and folded at once may be folded without building it; a fold or
// a build with side effects, or a tree used twice, must still be built.
class Node {
    public var left:Node;
    public var right:Node;
    public var value:Int;

    public function new(left, right, value) {
        this.left = left;
        this.right = right;
        this.value = value;
    }

    public function weigh():Int {
        if (left == null) return value * 3;
        return value + 2 * left.weigh() - right.weigh();
    }

    public function count():Int {
        Main.visits += 1;
        if (left == null) return 1;
        return 1 + left.count() + right.count();
    }
}

class Main {
    public static var visits = 0;
    public static var built = 0;

    static function grow(value:Int, depth:Int):Node {
        if (depth > 0)
            return new Node(grow(value * 2 + 1, depth - 1), grow(value - 3, depth - 1), value);
        return new Node(null, null, value);
    }

    static function growCounted(value:Int, depth:Int):Node {
        built += 1;
        if (depth > 0)
            return new Node(growCounted(value + 1, depth - 1), growCounted(value - 1, depth - 1), value);
        return new Node(null, null, value);
    }

    static function main() {
        // The same fold over a tree that is kept, so it is built.
        var kept = grow(7, 10);
        var expected = kept.weigh();
        if (grow(7, 10).weigh() != expected) throw "fused fold differs";
        if (kept.weigh() != expected) throw "kept tree changed";

        // A fold that counts its visits runs on every node.
        if (grow(1, 6).count() != 127 || visits != 127) throw "fold with side effects";

        // A build that counts its calls builds every node.
        var size = growCounted(0, 5).weigh();
        if (built != 63) throw "build with side effects";
        var again = growCounted(0, 5);
        if (again.weigh() != size) throw "build with side effects folded";

        Sys.println("CONFORMANCE_OK");
    }
}
