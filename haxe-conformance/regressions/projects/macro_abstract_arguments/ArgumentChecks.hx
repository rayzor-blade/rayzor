import haxe.macro.Context;
import haxe.macro.Expr;

class ArgumentChecks {
    public static macro function rejects(value:Expr):Expr {
        var rejected = try {
            Context.typeof(value);
            false;
        } catch (error:haxe.macro.Expr.Error) {
            true;
        };
        return macro $v{rejected};
    }
}
