// A `++` on the line after a block starts a new statement; a parameter
// incremented in a loop is loop-carried; `var x = null` takes its first
// assignment's type; `Null<String>` reaches a Dynamic slot as a box; a class
// whose `iterator()` returns `Iterator<T>` iterates, twice in one function;
// a `{ iterator: f }` structure iterates and types `Lambda.array`; a
// `catch (e:Dynamic)` sees the thrown value.
class Bag {
    public var items:Array<Int> = [1, 2, 3];
    public function new() {}
    public function iterator():Iterator<Int> return items.iterator();
}
class PostfixLoopsIterables {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function scan(str:String, p:Int):Int {
        var n = 0;
        while (p < str.length) {
            var c = str.charCodeAt(p);
            switch (c) {
                case 60: n++;
                default: p += 1;
            }
            ++p;
        }
        return p * 10 + n;
    }
    static function walk(str:String, p:Int):Int {
        while (p < str.length) {
            if (str.charCodeAt(p) == 60) return p;
            ++p;
        }
        return -1;
    }
    static function showD(x:Dynamic):String return Std.string(x);
    static function main() {
        var q = 0;
        if (q == 0) { q = 5; }
        ++q;
        check("postfix after block", Std.string(q), "6");
        check("switch in loop", Std.string(scan("<ab<", 0)), "42");
        check("param counter", Std.string(walk("ab<", 0)), "2");
        var name = null;
        var tmp;
        tmp = "hello";
        name = tmp;
        var m = new haxe.ds.StringMap<String>();
        m.set(name, "v");
        check("null local typed", Std.string(m.exists(name)), "true");
        check("Null<String> to Dynamic", showD(m.get("hello")), "v");
        var buf = new StringBuf();
        buf.add(m.get("hello"));
        check("Map.get into StringBuf", buf.toString(), "v");
        var b = new Bag();
        var s1 = 0;
        for (x in b) s1 += x;
        var s2 = 0;
        for (y in b) s2 += y * 10;
        check("iterator() handle", Std.string(s1), "6");
        check("second handle loop", Std.string(s2), "60");
        check("items survive", Std.string(b.items.length), "3");
        var arr = Lambda.array({ iterator: b.iterator });
        check("structural iterable", arr.join(","), "1,2,3");
        check("Lambda.count on class", Std.string(Lambda.count(b)), "3");
        var caught = "";
        try { throw "boom"; } catch (e:Dynamic) caught = Std.string(e);
        check("caught Dynamic", caught, "boom");
        var x = Xml.parse('<foo href="a">xx</foo> bar&lt;');
        check("xml", x.toString(), '<foo href="a">xx</foo> bar&lt;');
        check("xml attributes", Lambda.array({ iterator: x.firstElement().attributes }).join("#"), "href");
        Sys.println("CONFORMANCE_OK");
    }
}
