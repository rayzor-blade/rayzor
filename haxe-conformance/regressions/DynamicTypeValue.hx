class DynamicTypeValue {
    static function main() {
        var types:Array<Dynamic> = [null, Dynamic];
        if (types.length != 2) throw "type value omitted";
        if (types[0] == types[1]) throw "Dynamic equals null";
        if (Std.isOfType(0, types[0])) throw "null type target";
        if (!Std.isOfType(0, types[1])) throw "Dynamic type target";
        if (!Std.isOfType(0, Dynamic)) throw "direct Dynamic target";
        Sys.println("CONFORMANCE_OK");
    }
}
