// A Dynamic slot holding an erased primitive carries the value, not a box.
// Every reader below used to reach haxe_std_string_ptr, which dereferenced
// those bits as an address.
class DynamicElements {
    static function main() {
        var d:Array<Dynamic> = [];
        d.push(1); d.push(2);
        if (d[0] != 1) throw "index";
        var sum = 0;
        for (v in d) sum += v;
        if (sum != 3) throw "iteration";
        if (("x" + d[0]) != "x1") throw "concat";
        if (d.join(",") != "1,2") throw "join";
        if (Std.string(d[0]) != "1") throw "Std.string";
        var a:Dynamic = 3;
        var b:Dynamic = 4;
        if ((a + b) != 7) throw "arithmetic";
        var m = new Map<String,Dynamic>();
        m.set("k", 7);
        if (("" + m.get("k")) != "7") throw "map value";
        if (m.get("nope") != null) throw "missing key";
        trace("CONFORMANCE_OK");
    }
}
