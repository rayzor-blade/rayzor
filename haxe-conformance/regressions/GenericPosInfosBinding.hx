class GenericPosInfosBinding {
    static function second<T>(a:T, b:T, ?pos:haxe.PosInfos):String {
        return Std.string(b);
    }
    static function main() {
        var d:Dynamic = "test";
        if (second(d, "test") != "test") throw "dynamic and string";
        var obj = {bar: "x"};
        if (second(Reflect.field(obj, "bar"), "x") != "x") throw "reflected field";
        Sys.println("CONFORMANCE_OK");
    }
}
