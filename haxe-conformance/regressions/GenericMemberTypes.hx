private class GenericItem<T> {
    public var value:T;
    public function new(value:T) this.value = value;
    public function record() return {value: value};
    public function nested() return new GenericItem([value]);
    public function copy() return new GenericItem(value);
}

private class GenericItemIterator<T> {
    var values:Array<T>;
    var index:Int = 0;
    public function new(values:Array<T>) this.values = values;
    public function hasNext():Bool return index < values.length;
    public function next() return new GenericItem(values[index++]);
}

private class GenericRecordIterator<T> {
    var values:Array<T>;
    var index:Int = 0;
    public function new(values:Array<T>) this.values = values;
    public function hasNext():Bool return index < values.length;
    public function next() return {value: values[index++]};
}

private class GenericPair<A, B> {
    public var first:A;
    public var second:B;
    public function new(first:A, second:B) {
        this.first = first;
        this.second = second;
    }
    public function swapped():GenericPair<B, A> return new GenericPair(second, first);
}

private class ConstructorStringDefault {
    public var value:String;
    public function new(?value:String = "fallback") this.value = value;
}

class GenericMemberTypes {
    static function main() {
        var item = new GenericItem("hello");
        if ('${item.value}' != "hello") throw "generic field";
        if ('${item.record().value}' != "hello") throw "generic record return";
        if ('${item.nested().value[0]}' != "hello") throw "nested generic return";
        if ('${item.copy().value}' != "hello") throw "generic constructor forwarding";
        var callback = new GenericItem<Int->Int>(function(n:Int):Int return n * n);
        if (callback.value(3) != 9) throw "generic function field";
        var floatCallback = new GenericItem(function(n:Float):Float return n + 0.5);
        if (floatCallback.value(1.5) != 2.0) throw "inferred generic function field";
        var swapped = new GenericPair(7, "seven").swapped();
        if ('${swapped.first}:${swapped.second}' != "seven:7") throw "reordered type parameters";
        var text = "";
        for (entry in new GenericItemIterator(["a", "b"])) text += entry.value;
        if (text != "ab") throw "generic class iterator";
        text = "";
        for (entry in new GenericRecordIterator(["c", "d"])) text += entry.value;
        if (text != "cd") throw "generic record iterator";
        text = "";
        for (entry in new GenericItemIterator([2, 3])) text += entry.value;
        if (text != "23") throw "integer instantiation";
        text = "";
        for (entry in new GenericItemIterator([1.5, 2.5])) text += entry.value;
        if (text != "1.52.5") throw "float instantiation";
        var nullable = new GenericItem<String>(null);
        if ('${nullable.value}!' != "null!") throw "nullable generic string";
        var missing:String = null;
        if ("x" + missing != "xnull") throw "null concatenation rhs";
        if ("" + missing + missing != "nullnull") throw "null concatenation chain";
        if (new ConstructorStringDefault(null).value != "fallback") throw "constructor null default";
        if (new ConstructorStringDefault("").value != "") throw "constructor empty argument";
        if (new ConstructorStringDefault("given").value != "given") throw "constructor supplied argument";
        Sys.println("CONFORMANCE_OK");
    }
}
