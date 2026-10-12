private enum GeneratedValue<D, E> {
    Item(value:D);
    End(value:E);
}
private abstract ValueGenerator<D, E>(()->GeneratedValue<D, E>) {
    public function next():GeneratedValue<D, E> return this();
    @:from static function infinite<D, E>(f:()->D):ValueGenerator<D, E> {
        return function() return Item(f());
    }
}
class GenericEnumClosureParameters {
    static function main() {
        var count = 0;
        var ints:ValueGenerator<Int, String> = function():Int return count++;
        var first = ints.next();
        var second = ints.next();
        if (Std.string(first) != "Item(0)" || Std.string(second) != "Item(1)") throw "Int generator";
        var floats:ValueGenerator<Float, String> = function():Float return 1.5;
        var value = floats.next();
        if (Std.string(value) != "Item(1.5)") throw "Float generator";
        if (Type.enumParameters(value)[0] != 1.5) throw "Float parameter";
        switch (value) {
            case Item(number): if (number != 1.5) throw "typed Float parameter";
            case End(_): throw "unexpected end";
        }
        var strings:ValueGenerator<String, Int> = function():String return "hello";
        var text = strings.next();
        if (Std.string(text) != "Item(hello)" || Type.enumParameters(text)[0] != "hello") throw "String generator";
        Sys.println("CONFORMANCE_OK");
    }
}
