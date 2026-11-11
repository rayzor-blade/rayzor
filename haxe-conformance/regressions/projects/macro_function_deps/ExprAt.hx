#if macro
import haxe.macro.Expr;

class ExprAt {
	static public inline function at(e:ExprDef, ?pos:Position):Expr
		return {expr: e, pos: pos == null ? haxe.macro.Context.currentPos() : pos};
}
#end
