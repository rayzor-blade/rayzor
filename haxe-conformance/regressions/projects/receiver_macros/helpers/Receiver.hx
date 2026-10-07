package helpers;
import haxe.macro.Expr;
class Receiver {
    public var value:Int;
    public function new(value:Int) this.value = value;
    public function dupe():String return "ordinary";
    macro public function identity(ethis:Expr):Expr {
        var name = "receiverValue";
        var declaration = macro var $name = $ethis;
        var result = {expr: EConst(CIdent(name)), pos: haxe.macro.Context.currentPos()};
        return macro (function() { $declaration; return $result; })();
    }
    macro public function typeName(ethis:Expr):Expr {
        return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(ethis))};
    }
}
