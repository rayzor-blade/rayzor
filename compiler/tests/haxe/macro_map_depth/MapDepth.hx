class MapDepth {
    static var shared:Map<String, Int> = new Map();

    static function descend(map:Map<String, Int>, depth:Int) {
        if (depth > 0) descend(map, depth - 1);
        else map.set("answer", 42);
    }

    static function reader(map:Map<String, Int>):Void->Int {
        return function() return map.get("answer");
    }

    public static function check():Bool {
        var map = new Map<String, Int>();
        var alias = map;
        var holder = {map: map};
        var array = [map];
        var read = reader(map);
        descend(map, 32);
        if (alias.get("answer") != 42 || holder.map.get("answer") != 42
            || array[0].get("answer") != 42 || read() != 42) throw "helper aliases";
        holder.map.set("answer", 43);
        if (map.get("answer") != 43 || read() != 43) throw "nested receiver";
        var change = function() { map["answer"] = 44; };
        change();
        if (alias.get("answer") != 44 || read() != 44) throw "captured index write";
        var copy = map.copy();
        copy.set("answer", 99);
        if (map.get("answer") != 44 || copy.get("answer") != 99) throw "independent copy";
        if (!copy.remove("answer") || copy.remove("answer")) throw "remove result";
        if (map.get("answer") != 44) throw "remove alias";
        alias.clear();
        if (map.exists("answer") || holder.map.exists("answer")) throw "clear aliases";
        descend(shared, 4);
        if (shared.get("answer") != 42) throw "static alias";
        shared.clear();
        var key = {id: 1};
        var objects = new Map<{id:Int}, Int>();
        objects.set(key, 7);
        var keys = objects.copy();
        key.id = 2;
        if (keys.get(key) != 7 || objects.get(key) != 7
            || objects.exists({id: 2})) throw "object identity";
        var values = ["value" => {count: 1}];
        var shallow = values.copy();
        var value = shallow.get("value");
        value.count = 2;
        if (values.get("value").count != 2) throw "shallow copy";
        var ints = new haxe.ds.IntMap<String>();
        ints.set(3, "three");
        if (ints.get(3) != "three") throw "concrete map";
        return true;
    }

    public static function vmCheck():Bool {
        var map = new Map<String, Int>();
        var alias = map;
        var holder = {map: map};
        MapDepth.vmWrite(map);
        if (alias.get("answer") != 42 || holder.map.get("answer") != 42) return false;
        holder.map["answer"] = 43;
        if (alias.get("answer") != 43) return false;
        var copy = map.copy();
        copy.set("answer", 99);
        if (map.get("answer") != 43 || copy.get("answer") != 99) return false;
        if (!copy.remove("answer") || copy.remove("answer")) return false;
        alias.clear();
        return !map.exists("answer");
    }

    public static function vmWrite(map:Map<String, Int>) {
        map.set("answer", 42);
    }
}
