#if macro
import haxe.macro.Context;
import haxe.macro.Expr;

class ChainMac {
	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		if (Context.getLocalClass().get().isInterface)
			return fields;
		var name = Context.getLocalClass().get().name;
		fields.push({name: "TAG", access: [APublic, AStatic], kind: FVar(macro :String, macro $v{name}), pos: Context.currentPos()});
		return fields;
	}
}
#end
