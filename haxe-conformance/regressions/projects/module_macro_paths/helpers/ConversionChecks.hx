package helpers;

import haxe.macro.Context;
import haxe.macro.Expr;
import haxe.macro.TypeTools;

class MacroChecks {
    public static macro function describe(value:Expr):Expr {
        var name = TypeTools.toString(Context.typeof(value));
        return macro $v{name};
    }

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
