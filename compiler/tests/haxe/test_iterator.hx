class Range {
    var current:Int;
    var end:Int;

    public function new(s:Int, e:Int) {
        current = s;
        end = e;
    }

    public function hasNext():Bool {
        return current < end;
    }

    public function next():Int {
        var v = current;
        current = current + 1;
        return v;
    }

    public function iterator():Range {
        return this;
    }
}

class Main {
    static function main() {
        var sum = 0;
        for (x in new Range(1, 6)) {
            sum = sum + x;
        }
        trace(sum); // 15

        // Test with explicit iterator call
        var r = new Range(10, 13);
        var sum2 = 0;
        for (v in r) {
            sum2 = sum2 + v;
        }
        trace(sum2); // 33 (10+11+12)

        trace("done");
    }
}
