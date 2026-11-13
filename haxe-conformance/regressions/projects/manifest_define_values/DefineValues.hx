import haxe.macro.Context;
import haxe.macro.Expr;

class DefineValues {
	public static macro function value(name:String):Expr {
		var v = Context.definedValue(name);
		return macro $v{v == null ? "<null>" : v};
	}

	public static macro function defined(name:String):Expr
		return macro $v{Context.defined(name)};
}
