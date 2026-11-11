#if macro
import haxe.macro.Context;
import haxe.macro.Expr;

class AbstractBuildMac {
	/** Adds `static var TAG_<name> = name`. **/
	public static function tag(name:String):Array<Field> {
		var fields = Context.getBuildFields();
		fields.push({name: "TAG_" + name, access: [APublic, AStatic], kind: FVar(macro :String, macro $v{name}), pos: Context.currentPos()});
		return fields;
	}

	/** Adds `of(set)`, reading the field of `record` each value names; needs the typer. **/
	public static function keys(record:String):Array<Field> {
		var fields = Context.getBuildFields();
		var pos = Context.currentPos();
		var names = switch Context.follow(Context.getType(record)) {
			case TAnonymous(a): [for (f in a.get().fields) f.name];
			case _: Context.error('$record is not a record', pos);
		}
		var cases:Array<Case> = [];
		for (f in fields)
			switch f.kind {
				case FVar(_, {expr: EConst(CString(name))}) if (names.indexOf(name) >= 0):
					cases.push({values: [macro $v{name}], expr: macro set.$name});
				case _:
			}
		var body:Expr = {expr: ESwitch(macro this, cases, macro throw "no key"), pos: pos};
		fields.push({
			name: "of",
			access: [APublic],
			kind: FFun({args: [{name: "set", type: macro :AbstractBuildMacros.Rec}], ret: macro :Int, expr: macro return $body}),
			pos: pos
		});
		return fields;
	}
}
#end
