private enum PrintedKey {
    Item(number:Int, text:String);
}

class GenericEnumMapPrinting {
    static function render<K:EnumValue, V>(map:haxe.ds.EnumValueMap<K, V>):String {
        return map.toString();
    }

    static function renderTree<K, V>(tree:haxe.ds.BalancedTree<K, V>):String {
        return tree.toString();
    }

    static function main() {
        var map = new Map<PrintedKey, Int>();
        map.set(Item(0, "foo"), 1);
        if (map.toString() != "[Item(0,foo) => 1]") throw "enum map printing";

        var enums = new haxe.ds.EnumValueMap<PrintedKey, String>();
        enums.set(Item(2, "key"), "value");
        if (render(enums) != "[Item(2,key) => value]") throw "generic enum receiver";

        var floats = new haxe.ds.BalancedTree<String, Float>();
        floats.set("float", 1.5);
        if (renderTree(floats) != "[float => 1.5]") throw "generic float value";
        var booleans = new haxe.ds.BalancedTree<Int, Bool>();
        booleans.set(0, false);
        if (renderTree(booleans) != "[0 => false]") throw "generic integer and boolean";
        Sys.println("CONFORMANCE_OK");
    }
}
