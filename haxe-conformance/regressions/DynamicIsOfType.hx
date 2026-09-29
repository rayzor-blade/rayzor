class DynamicIsOfType {
    static function main() {
        var intType:Dynamic = Int;
        if (!Std.isOfType(2, intType)) throw "Int target";
        if (Std.isOfType("two", intType)) throw "Int rejects String";

        var types:Array<Dynamic> = [Int, String, Array];
        if (!Std.isOfType(3, types[0])) throw "boxed Int target";
        if (!Std.isOfType("three", types[1])) throw "boxed String target";
        if (!Std.isOfType([], types[2])) throw "boxed Array target";
        if (Std.isOfType(null, types[0])) throw "null rejects Int";

        var baseType:Dynamic = IsBase;
        var childType:Dynamic = IsChild;
        if (!Std.isOfType(new IsChild(), baseType)) throw "class hierarchy";
        if (Std.isOfType(new IsBase(), childType)) throw "reject parent as child";
        Sys.println("CONFORMANCE_OK");
    }
}

private class IsBase {
    public function new() {}
}

private class IsChild extends IsBase {
    public function new() { super(); }
}
