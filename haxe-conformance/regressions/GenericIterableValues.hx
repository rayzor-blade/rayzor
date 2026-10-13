private enum IterValue { Item(value:String); }
class GenericIterableValues {
    static function first<T>(values:Iterable<T>):T {
        for (value in values) return value;
        throw "empty";
    }
    static function equal<T>(values:Iterable<T>, expected:T):Bool {
        for (value in values) if (value == expected) return true;
        return false;
    }
    static function main() {
        var word = ["fi", "eld"].join("");
        if (!Lambda.has([word], "field")) throw "Lambda.has string";
        if (!equal([word], "field")) throw "generic string comparison";
        if (first([word]) != "field") throw "generic string return";
        if (first([1.25]) != 1.25 || !equal([1.25], 1.25)) throw "generic float";
        if (first([42]) != 42 || !equal([42], 42)) throw "generic integer";
        var entry = Item("enum");
        if (first([entry]) != entry) throw "generic enum identity";
        var dynamicValues:Array<Dynamic> = [word, 1.25, entry];
        if (first(dynamicValues) != "field") throw "generic Dynamic";
        dynamicValues.push(42);
        if (!equal(dynamicValues, "field") || !equal(dynamicValues, 42)) throw "mixed Dynamic comparison";
        if (!Lambda.has(dynamicValues, "field") || !Lambda.has(dynamicValues, 1.25) || !Lambda.has(dynamicValues, 42)) throw "mixed Dynamic Lambda.has";
        var values = new List<String>();
        values.add(word);
        if (!equal(values, "field") || first(values) != "field") throw "generic list";
        trace("CONFORMANCE_OK");
    }
}
