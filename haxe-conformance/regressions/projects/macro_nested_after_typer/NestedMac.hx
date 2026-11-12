import haxe.macro.Context;
import haxe.macro.Expr;

class NestedMac {
	public static macro function wrap(e:Expr):Expr {
		return macro "[" + $e + "]";
	}

	/** Asks the typer a question before it returns, as hxx does. **/
	public static macro function outerTyped(e:Expr):Expr {
		var s = switch (e.expr) {
			case EConst(CString(s)): s;
			case _: Context.error("a string", e.pos);
		}
		Context.getType("String");
		var parsed = Context.parseInlineString(s, e.pos);
		return macro "<" + $parsed + ">";
	}
}
