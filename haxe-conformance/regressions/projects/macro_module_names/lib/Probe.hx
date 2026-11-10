package lib;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;

class Probe {
	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		var c = Context.getLocalClass().get();
		var parents = [for (i in c.interfaces) i.t.get().name];
		parents.sort(Reflect.compare);
		var parts = [
			"iface=" + c.isInterface + " super=" + (c.superClass == null ? "none" : c.superClass.t.get().name) + " ifaces="
			+ parents.join(","),
			"alpha=" + Defs.Kind.Alpha + " twice=" + Defs.Util.twice(4) + " limit=" + Defs.Util.LIMIT,
			"all=" + Defs.Kind.createAll().join(","),
		];
		var b = Defs.Kind.Beta(5);
		parts.push("name=" + b.getName() + " index=" + b.getIndex() + " eq=" + b.equals(Defs.Kind.Beta(5)));
		var f = Defs.Type.TFun([1, 2]);
		parts.push("tfun=" + f.getName() + ":" + f.getParameters()[0].length + " tint=" + Defs.Type.TInt);
		fields.push({
			name: "REPORT",
			access: [APublic, AStatic],
			kind: FVar(macro :String, macro $v{parts.join(" | ")}),
			pos: Context.currentPos()
		});
		return fields;
	}
}
#end
