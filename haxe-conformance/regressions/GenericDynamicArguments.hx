class GenericDynamicArguments {
    macro static function typeName(e:haxe.macro.Expr) {
        return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(e))};
    }
    static function same<T>(a:T, b:T):Bool return a == b;
    static function show<T>(a:T, b:T):String return Std.string(a) + "/" + Std.string(b);
    static function pick<T>(a:T, b:T):T return a;
    static function main() {
        var number:Dynamic = 1.5;
        if (!same(number, 1.5) || !same(1.5, number)) throw "dynamic float arguments";
        var text:Dynamic = "0";
        if (same(text, 0.0) || same(0.0, text)) throw "dynamic tag preservation";
        if (show(text, 0.0) != "0/0") throw "dynamic string arguments";
        if (typeName(GenericDynamicArguments.pick(text, 0.0)) != "Float") throw "generic result inference";
        Sys.println("CONFORMANCE_OK");
    }
}
