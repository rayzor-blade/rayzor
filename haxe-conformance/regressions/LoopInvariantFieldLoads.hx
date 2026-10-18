// Field reads lifted out of a loop must still see every write the loop makes:
// to the same field through an alias, through a call, or by replacing the
// object the field is read from. A read guarded by a null check stays guarded.
class Box {
    public var v:Int;
    public var w:Int;

    public function new(v:Int) {
        this.v = v;
        this.w = 0;
    }
}

class Holder {
    public var items:Array<Float>;
    public var scale:Float;
    public var count:Int;
    public var cur:Box;

    public function new() {
        items = [1.0, 4.0, 9.0, 16.0];
        scale = 2.0;
        count = 0;
        cur = new Box(1);
    }

    function bump():Void
        count += 3;

    public function nested():Float {
        var sum = 0.0;
        for (i in 0...items.length)
            for (j in i + 1...items.length) {
                var d = items[i] - items[j];
                sum += Math.sqrt(d * d) * scale;
                items[j] = items[j] + 1.0;
            }
        return sum;
    }

    public function viaCall():Int {
        var total = 0;
        for (i in 0...4) {
            total += count;
            bump();
        }
        return total;
    }

    public function replaced():Int {
        var total = 0;
        for (i in 0...4) {
            total += cur.v;
            cur = new Box(cur.v * 2);
        }
        return total;
    }
}

class Main {
    static function aliased(a:Box, b:Box):Int {
        var total = 0;
        for (i in 0...4) {
            total += a.v;
            b.v = b.v + 10;
            a.w = i;
        }
        return total;
    }

    static function guarded(o:Box, n:Int):Int {
        var total = 0;
        for (i in 0...n)
            if (o != null)
                total += o.v;
        return total;
    }

    static function main() {
        var h = new Holder();
        // pairs over [1,4,9,16] with later items bumped by one per earlier i
        if (h.nested() != 2.0 * (3 + 8 + 15 + 5 + 12 + 7)) throw "nested " + h.nested();
        if (h.viaCall() != 0 + 3 + 6 + 9) throw "call";
        if (h.replaced() != 1 + 2 + 4 + 8) throw "replaced";
        var box = new Box(1);
        if (aliased(box, box) != 1 + 11 + 21 + 31) throw "alias";
        if (box.w != 3) throw "w";
        if (aliased(new Box(5), new Box(0)) != 20) throw "distinct";
        if (guarded(null, 5) != 0) throw "null";
        if (guarded(new Box(2), 5) != 10) throw "guarded";
        Sys.println("CONFORMANCE_OK");
    }
}
