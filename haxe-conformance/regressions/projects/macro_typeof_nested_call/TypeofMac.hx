import haxe.macro.Context;
import haxe.macro.Expr;
import haxe.macro.TypeTools;

class TypeofMac {
	public static macro function wrap(e:Expr):Expr {
		return macro [$e];
	}

	public static macro function typeName(e:Expr):Expr {
		var t = try Context.typeof(e) catch (_:Dynamic) null;
		return macro $v{t == null ? "null" : TypeTools.toString(t)};
	}

	public static macro function typeNameOf(e:Expr):Expr {
		var s = switch (e.expr) {
			case EConst(CString(s)): s;
			case _: Context.error("a string", e.pos);
		}
		var parsed = Context.parseInlineString(s, e.pos);
		var t = try Context.typeof(parsed) catch (_:Dynamic) null;
		return macro $v{t == null ? "null" : TypeTools.toString(t)};
	}
}
