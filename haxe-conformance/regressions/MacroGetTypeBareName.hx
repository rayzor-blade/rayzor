import haxe.io.Input;

// Context.getType finds a bare name through the module's imports, its package
// or the top level, not a loaded packaged type of that name.
class MacroGetTypeBareName {
	public static macro function probe(name:String):haxe.macro.Expr {
		var found = try haxe.macro.TypeTools.toString(haxe.macro.Context.getType(name)) catch (_:Dynamic) "none";
		return macro $v{found};
	}

	static function main() {
		var got = [probe("Input"), probe("Output"), probe("Bytes"), probe("haxe.io.Output"), probe("String")].join(",");
		if (got != "haxe.io.Input,none,none,haxe.io.Output,String")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
