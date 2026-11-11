import haxe.macro.Context;
import haxe.macro.Expr;

class UnifyMac {
	public static macro function u(e:Expr, target:String):Expr {
		return macro $v{Context.unify(Context.typeof(e), Context.getType("Kinds." + target))};
	}
}
