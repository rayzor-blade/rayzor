class DynamicArrayIteration {
    static function collect(array:Dynamic):Array<Dynamic> {
        var iterator:Iterator<Dynamic> = array.iterator();
        var result:Array<Dynamic> = [];
        for (value in iterator) result.push(value);
        return result;
    }

    static function main() {
        var ints = collect([0, -2, 1]);
        if (ints.length != 3 || ints[0] != 0 || ints[1] != -2 || ints[2] != 1) throw "erased integers";
        if (Std.string(ints[0]) != "0" || !Std.isOfType(ints[0], Int)) throw "integer tag";
        var floats = collect([0.0, 1.5, -2.25]);
        if (floats[0] != 0.0 || floats[1] != 1.5 || floats[2] != -2.25) throw "erased floats";
        var booleans = collect([false, true]);
        if (booleans[0] != false || booleans[1] != true || !Std.isOfType(booleans[0], Bool)) throw "erased booleans";
        var strings = collect(["hello", "world"]);
        if (strings[0] != "hello" || strings[1] != "world") throw "erased strings";
        var nested = collect([[0, 1], [2]]);
        var first:Array<Int> = nested[0];
        if (first[0] != 0 || first[1] != 1) throw "nested array layout";
        var mixed:Array<Dynamic> = [0, "text", 1.5, false, null];
        var copied = collect(mixed);
        if (copied[0] != 0 || copied[1] != "text" || copied[2] != 1.5 || copied[3] != false || copied[4] != null) throw "boxed slots";
        var item = new IteratorItem();
        var objects = collect([item]);
        if (objects[0] != item || !Std.isOfType(objects[0], IteratorItem)) throw "class identity";
        if (collect([]).length != 0) throw "empty iterator";

        var reflected:Dynamic = {values: [0, 2]};
        if (collect(Reflect.field(reflected, "values"))[0] != 0) throw "reflected array layout";
        Reflect.setField(reflected, "extra", true);
        if (collect(Reflect.field(reflected, "values"))[1] != 2) throw "promoted field layout";
        if (!Reflect.deleteField(reflected, "extra") || Reflect.hasField(reflected, "extra")) throw "delete boxed field";
        if (collect(Reflect.field(reflected, "values"))[1] != 2) throw "deleted field layout";
        Reflect.setField(reflected, "values", [1.5, 2.25]);
        if (collect(Reflect.field(reflected, "values"))[0] != 1.5) throw "replaced field layout";
        var inlineObject:Dynamic = {values: [0]};
        Reflect.setField(inlineObject, "values", [3.5]);
        if (collect(Reflect.field(inlineObject, "values"))[0] != 3.5) throw "retyped field layout";
        var deletion:Dynamic = {values: [0, 4], extra: true};
        if (!Reflect.deleteField(deletion, "extra") || Reflect.deleteField(deletion, "missing")) throw "delete inline field";
        if (collect(Reflect.field(deletion, "values"))[1] != 4) throw "deleted inline layout";

        var source = [0];
        var erased:Dynamic = source;
        var iterator:Iterator<Dynamic> = erased.iterator();
        if (!iterator.hasNext() || iterator.next() != 0 || iterator.hasNext()) throw "iterator exhaustion";
        var typed = [0, 2].iterator();
        if (typed.next() != 0 || typed.next() != 2 || typed.hasNext()) throw "typed iterator";
        var typedFloats = [1.5].iterator();
        if (typedFloats.next() != 1.5) throw "typed float iterator";
        var typedStrings = ["raw"].iterator();
        if (typedStrings.next() != "raw") throw "typed String iterator";

        var template = new haxe.Template("::foreach values::::__current__::::end::");
        if (template.execute({values: [0, 1]}) != "01") throw "template integers";
        if (template.execute({values: ["hello", "world"]}) != "helloworld") throw "template strings";
        Sys.println("CONFORMANCE_OK");
    }
}

class IteratorItem {
    public function new() {}
}
