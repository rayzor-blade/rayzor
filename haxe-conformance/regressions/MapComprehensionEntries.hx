class MapComprehensionEntries {
    static var keyCalls = 0;
    static var valueCalls = 0;
    static function key(value:Int):Int { keyCalls++; return value; }
    static function value(number:Int):Int { valueCalls++; return number * 2; }

    static function checkMaps<K, V>(left:Map<K, V>, right:Map<K, V>) {
        for (key in left.keys()) if (left[key] != right[key]) throw "generic map indexing";
        for (key in right.keys()) if (left[key] != right[key]) throw "generic map indexing";
    }

    static macro function macroCount() {
        var map = [for (i in 0...4) if (i > 1) (i => i * 2)];
        var count = 0;
        for (key in map.keys()) count++;
        return macro $v{count};
    }

    static function main() {
        var grouped = [for (i in 0...3) (i => i * 2)];
        if (grouped[0] != 0 || grouped[2] != 4) throw "parenthesized entry";
        checkMaps(grouped, [0 => 0, 1 => 2, 2 => 4]);
        var filtered = [for (i in 0...4) if (i > 0) if (i < 3) key(i) => value(i)];
        if (filtered.exists(0) || filtered.exists(3) || filtered[1] != 2 || filtered[2] != 4) throw "filtered entries";
        if (keyCalls != 2 || valueCalls != 2) throw "filtered side effects";
        var empty = [for (i in 0...2) if (false) key(i) => value(i)];
        if (empty.exists(0) || keyCalls != 2 || valueCalls != 2) throw "empty filter";
        var nested = [for (i in 0...2) for (j in 0...2) if (i == j) (i => j + 5)];
        if (nested[0] != 5 || nested[1] != 6) throw "nested loops";
        var source = ["first" => 1, "second" => 2];
        var copied = [for (name => number in source) if (number > 1) name => number * 3];
        if (copied.exists("first") || copied["second"] != 6) throw "key-value iteration";
        var conditional = [for (i in 0...3) (i == 0 ? 10 : i) => i];
        if (conditional[10] != 0 || conditional[2] != 2) throw "conditional key";
        var object = new MapEntryKey();
        var objects = [for (i in 0...1) object => 7];
        if (objects[object] != 7) throw "object map allocation";
        if (macroCount() != 2) throw "macro comprehension";
        Sys.println("CONFORMANCE_OK");
    }
}

class MapEntryKey {
    public function new() {}
}
