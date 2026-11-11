import haxe.macro.Context;
import haxe.macro.Expr;

using ExprAt;

class DepsMac {
	// the extension called on a constructor application
	public static macro function onCtor():Expr
		return EConst(CString("ctor")).at(Context.currentPos());

	// on a variable of the enum type
	public static macro function onVar():Expr {
		var def:ExprDef = EConst(CString("var"));
		return def.at(Context.currentPos());
	}

	// on the enum value built by a call
	public static macro function onField():Expr
		return EObjectDecl([{field: "a", expr: macro 1}]).at(Context.currentPos());
}
