package lib;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
import lib.Alloc;

class Mac {
	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		var a = new Alloc(5);
		var text = "sum=" + Use.sum(a) + " push=" + Use.push(a, 7) + " sum2=" + Use.sum(a) + " tag=" + a.tagged;
		fields.push({
			name: "TEXT",
			access: [APublic, AStatic],
			kind: FVar(macro :String, macro $v{text}),
			pos: Context.currentPos()
		});
		return fields;
	}
}
#end
