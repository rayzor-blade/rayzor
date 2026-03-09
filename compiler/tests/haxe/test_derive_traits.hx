@:derive([PartialEq, Eq, PartialOrd, Ord, Hash])
class Point {
    public var x:Int;
    public var y:Int;

    public function new(x:Int, y:Int) {
        this.x = x;
        this.y = y;
    }
}

class Main {
    static function main() {
        var a = new Point(3, 4);
        var b = new Point(3, 4);
        var c = new Point(1, 2);
        var d = new Point(3, 5);

        // PartialEq: field-by-field ==
        trace(a == b);  // true (same field values)
        trace(a == c);  // false (different values)
        trace(a != c);  // true
        trace(a != b);  // false

        // Same object
        trace(a == a);  // true (pointer equality fast path)

        // PartialOrd: lexicographic comparison
        trace(c < a);   // true (1 < 3)
        trace(a < d);   // true (x equal, 4 < 5)
        trace(a > c);   // true
        trace(a <= b);  // true (equal)
        trace(a >= b);  // true (equal)
        trace(a < b);   // false (equal)
        trace(a > b);   // false (equal)

        // Hash: hashCode()
        var h1 = a.hashCode();
        var h2 = b.hashCode();
        trace(h1 == h2);  // true (same fields → same hash)
        trace(h1 != 0);   // true (non-trivial hash)
    }
}
