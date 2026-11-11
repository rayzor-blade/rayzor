package w;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
import w.Use;

class Mac {
	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		fields.push({
			name: "TEXT",
			access: [APublic, AStatic],
			kind: FVar(macro :String, macro $v{Use.run()}),
			pos: Context.currentPos()
		});
		return fields;
	}
}
#end
