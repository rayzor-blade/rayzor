// A macro that waits for the typer and one that runs at once still see their
// class's macro-time statics in source order; a `@:const` type argument is
// kept, and TypeTools.toComplexType gives the syntax of a typed type.
private typedef Sized<@:const L> = String;

class MacroOrderAndConstTypeArgs {
	#if macro
	static var stored:haxe.macro.Expr;
	#end

	static macro function store(e) {
		stored = haxe.macro.Context.storeTypedExpr(haxe.macro.Context.typeExpr(e));
		return macro null;
	}

	static macro function load() return stored;

	static macro function typeName(e:haxe.macro.Expr) {
		return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(e))};
	}

	static macro function constArg() {
		var ct = haxe.macro.TypeTools.toComplexType(haxe.macro.Context.typeof(macro (null : Sized<12>)));
		return macro $v{ct.match(TPath({name: "Sized", params: [TPExpr({expr: EConst(CInt("12"))})]}))};
	}

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		store(function(a:Array<Int>) return a[0] + a[1]);
		check("stored lambda", load()([3, 4]), 7);
		check("const argument", typeName((null : Sized<12>)), "Sized<12>");
		check("toComplexType", constArg(), true);
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
