package helpers;
import haxe.macro.Expr;
abstract MacroNumber<T>(Int) {
    public function new(value:Int) this = value;
    macro public function typeName(ethis:Expr):Expr {
        return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(ethis))};
    }
}
