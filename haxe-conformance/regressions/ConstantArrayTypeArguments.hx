typedef ArrayAsset<@:const T> = String;

class ConstantArrayTypeArguments {
    static macro function typeName(expression:haxe.macro.Expr) {
        return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(expression))};
    }

    static function main() {
        var first = typeName((null : ArrayAsset<["test", 1]>));
        if (first != "ArrayAsset<[\"test\", 1]>") throw "array argument: " + first;
        var second = typeName((null : ArrayAsset<["other", 2]>));
        if (second != "ArrayAsset<[\"other\", 2]>" || first == second) throw "distinct array arguments";
        Sys.println("CONFORMANCE_OK");
    }
}
