import haxe.ds.HashMap;

private class IterationKey {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    public function hashCode():Int { return id; }
}

class GenericHashMapIteration {
    static function main() {
        var words = new HashMap<IterationKey, String>();
        words.set(new IterationKey(1), "first");
        words.set(new IterationKey(2), "second");
        var count = 0;
        for (key => value in words) {
            if (value != (key.id == 1 ? "first" : "second")) throw "string iterator binding";
            count++;
        }
        if (count != 2) throw "string iteration count";
        var numbers = new HashMap<IterationKey, Float>();
        numbers.set(new IterationKey(3), 1.25);
        numbers.set(new IterationKey(4), -2.5);
        count = 0;
        for (key => value in numbers) {
            if (value != (key.id == 3 ? 1.25 : -2.5)) throw "float iterator binding";
            count++;
        }
        if (count != 2) throw "float iteration count";
        Sys.println("CONFORMANCE_OK");
    }
}
