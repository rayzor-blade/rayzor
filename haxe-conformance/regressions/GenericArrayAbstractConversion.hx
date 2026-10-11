private abstract IndexedValues<T>(haxe.ds.StringMap<T>) {
    private function new() this = new haxe.ds.StringMap<T>();
    public function get(key:String):Null<T> return this.get(key);
    function put(key:String, value:T) this.set(key, value);
    @:from public static function fromArray<K>(values:Array<K>) {
        var result = new IndexedValues();
        var index = 0;
        while (index < values.length) {
            var key = values[index++];
            var value = values[index++];
            result.put("k" + key, value);
        }
        return result;
    }
}
class GenericArrayAbstractConversion {
    static function main() {
        var values:IndexedValues<Int> = [1, 2, 3, 4];
        if (values.get("k1") != 2 || values.get("k3") != 4) throw "generic array conversion";
        var strings:IndexedValues<String> = ["a", "first", "b", "second"];
        if (strings.get("ka") != "first" || strings.get("kb") != "second") throw "string array conversion";
        var floats:IndexedValues<Float> = [1.0, 2.5, 3.0, 4.75];
        if (floats.get("k1") != 2.5 || floats.get("k3") != 4.75) throw "float array conversion";
        Sys.println("CONFORMANCE_OK");
    }
}
