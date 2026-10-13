enum MapEntry {
    Entry(name:String, value:String);
}

class MapInferredInLoop {
    static function main() {
        var entries = [{name: "one", value: "first"}, {name: "two", value: "second"}];
        var indexed = new Map();
        for (entry in entries) indexed[entry.name] = entry.value;
        if (indexed["one"] != "first" || indexed["two"] != "second") throw "index write";

        final methods = new Map();
        for (entry in entries) methods.set(entry.name, entry.value);
        if (methods.get("two") != "second") throw "method write";

        var matched = new Map();
        for (entry in [Entry("three", "third")]) switch entry {
            case Entry(key, value): matched[key] = value;
        }
        if (matched["three"] != "third") throw "pattern write";
        trace("CONFORMANCE_OK");
    }
}
