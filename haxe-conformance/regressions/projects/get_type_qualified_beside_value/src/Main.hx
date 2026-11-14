enum abstract Cursor(Int) {
	var Default = 0;
	var Progress = 1;
}

// p.Progress, not yet compiled, is that class; not the value `Progress`.
class Main {
	public static macro function probe(name:String):haxe.macro.Expr {
		var found = try switch haxe.macro.Context.getType(name) {
			case TInst(c, _): "class " + c.toString();
			case t: haxe.macro.TypeTools.toString(t);
		} catch (_:Dynamic) "none";
		return macro $v{found};
	}

	static function main() {
		var cursor:Cursor = Progress;
		var got = probe("Progress") + "," + probe("p.Progress") + " " + (cursor == Progress);
		if (got != "none,class p.Progress true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
