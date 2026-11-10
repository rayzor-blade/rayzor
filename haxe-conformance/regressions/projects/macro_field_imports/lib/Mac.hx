package lib;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
import lib.User;

class Mac {
	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		fields.push({
			name: "TEXT",
			access: [APublic, AStatic],
			kind: FVar(macro :String, macro $v{User.run()}),
			pos: Context.currentPos()
		});
		return fields;
	}
}
#end
