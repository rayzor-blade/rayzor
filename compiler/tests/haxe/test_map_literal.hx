import haxe.ds.StringMap;

class Main {
    static function main() {
        var m = new StringMap();
        m.set("hello", 42);
        trace(m);
        var val = m.get("hello");
        trace(val);
    }
}
