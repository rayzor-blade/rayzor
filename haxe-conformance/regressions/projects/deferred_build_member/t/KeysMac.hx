package t;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;

class KeysMac {
	public static function build(record:String):Array<Field> {
		var fields = Context.getBuildFields();
		var pos = Context.currentPos();
		var cases:Array<Case> = [];
		for (f in fields)
			switch f.kind {
				case FVar(_, {expr: EConst(CString(name))}):
					cases.push({values: [macro $v{name}], expr: macro set.$name});
				case _:
			}
		var recordType = Context.toComplexType(Context.getType(record));
		var body:Expr = {expr: ESwitch(macro this, cases, macro throw 'no key ' + this), pos: pos};
		fields.push({
			name: "of",
			access: [APublic],
			kind: FFun({args: [{name: "set", type: recordType}], ret: macro :Float, expr: macro return $body}),
			pos: pos
		});
		return fields;
	}
}
#end
