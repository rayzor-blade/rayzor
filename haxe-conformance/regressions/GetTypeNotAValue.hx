enum abstract Cursor(Int) {
	var Default = 0;
	var Progress = 1;
}

// Context.getType of a name that is only an enum-abstract value finds no type.
class GetTypeNotAValue {
	public static macro function probe(name:String):haxe.macro.Expr {
		var found = try haxe.macro.TypeTools.toString(haxe.macro.Context.getType(name)) catch (_:Dynamic) "none";
		return macro $v{found};
	}

	static function main() {
		var cursor:Cursor = Progress;
		var got = [probe("Progress"), probe("Cursor"), probe("Dynamic")].join(",") + " " + (cursor == Progress);
		if (got != "none,Cursor,Dynamic true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
