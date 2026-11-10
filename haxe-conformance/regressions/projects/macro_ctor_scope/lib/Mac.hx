package lib;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
import lib.Ast;

class Mac {
	public static function report():String {
		var funs:Array<Fun> = [{name: "f", kind: Fragment}];
		return [
			"has=" + Comp.hasStage(funs),
			"desc=" + Comp.describe(Main),
			"first=" + Link.first(),
			"kind=" + Link.kindIs(Kind.Vertex) + "," + Link.kindIs(Kind.Main),
		].join(" ");
	}

	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		fields.push({
			name: "TEXT",
			access: [APublic, AStatic],
			kind: FVar(macro :String, macro $v{report()}),
			pos: Context.currentPos()
		});
		return fields;
	}
}
#end
