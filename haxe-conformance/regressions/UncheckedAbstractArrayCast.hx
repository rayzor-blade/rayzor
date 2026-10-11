@:arrayAccess
private abstract ArrayView<T>(Array<T>) {
    public function new():Void this = new Array<T>();
    public function push(value:T):Int return this.push(value);
    @:from public static inline function fromArray<T,U>(values:Array<U>):ArrayView<T> {
        return cast values;
    }
    @:to public function toArray<T>():Array<T> {
        return cast this;
    }
}

class UncheckedAbstractArrayCast {
    public function new() {}
    function check() {
        var ints:ArrayView<Int> = [1, 2, 3];
        for (value in 0...2) ints.push(value);
        if (ints[0] != 1 || ints[1] != 2 || ints[2] != 3 || ints[3] != 0 || ints[4] != 1)
            throw "int array view";
        var floats:ArrayView<Float> = [1.5, 2.75];
        floats.push(3.25);
        if (floats[0] != 1.5 || floats[1] != 2.75 || floats[2] != 3.25)
            throw "float array view";
        var strings:ArrayView<String> = ["first", "second"];
        strings.push("third");
        if (strings[0] != "first" || strings[1] != "second" || strings[2] != "third")
            throw "string array view";
    }
    static function main() {
        new UncheckedAbstractArrayCast().check();
        Sys.println("CONFORMANCE_OK");
    }
}
