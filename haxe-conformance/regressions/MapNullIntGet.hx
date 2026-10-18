// `m.get(k)` of a map of Ints read as Null<Int> distinguishes an absent key
// from a stored 0, for String and Int keys alike.
class Main {
    static function main() {
        var m = new Map<String, Int>();
        m.set("a", 0);
        m.set("b", -5);
        m.set("c", 2147483647);
        var out:Array<Dynamic> = [];
        for (k in ["a", "b", "c", "zz"]) {
            var v = m.get(k);
            out.push(v == null ? "null" : Std.string(v));
        }
        var im = new Map<Int, Int>();
        im.set(1, -1);
        im.set(-7, 0);
        for (k in [1, -7, 3]) {
            var v = im.get(k);
            out.push(v == null ? "null" : Std.string(v));
        }
        var counts = new Map<String, Int>();
        for (w in ["x", "y", "x", "x"]) {
            var c = counts.get(w);
            counts.set(w, c == null ? 1 : c + 1);
        }
        out.push(counts.get("x"));
        out.push(counts.get("y"));
        if (out.join(",") != "0,-5,2147483647,null,-1,0,null,3,1") throw out.join(",");
        Sys.println("CONFORMANCE_OK");
    }
}
