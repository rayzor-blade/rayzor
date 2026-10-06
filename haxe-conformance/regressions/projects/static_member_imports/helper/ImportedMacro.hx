package helper;
import haxe.macro.Context;
import haxe.macro.Expr;
class ImportedMacro {
    public static macro function value(expr:Expr):Expr {
        Context.typeof(expr);
        return macro 7;
    }
}
