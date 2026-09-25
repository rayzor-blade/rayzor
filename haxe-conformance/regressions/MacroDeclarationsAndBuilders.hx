// A macro declared without a body outside `#if macro` takes its body from the
// macro side; MacroStringTools.formatString reads a format string; a rest
// parameter of expressions converts with toArray; an ordinary static
// function may serve as a @:build macro.
#if !macro
@:build(Builder.build())
class Built {
	public function new() {}
}

class MacroDeclarationsAndBuilders {
	static macro function format(s:String):haxe.macro.Expr;
	static macro function names(...exprs:haxe.macro.Expr):haxe.macro.Expr;

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		var w = 7;
		check("formatString", format('w=$w'), "w=7");
		check("rest toArray", names(a, 1, "s").join(","), 'a,1,"s"');
		check("static builder", Built.tag, "built");
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
#else
class MacroDeclarationsAndBuilders {
	static macro function format(s:String):haxe.macro.Expr {
		return haxe.macro.MacroStringTools.formatString(s, haxe.macro.Context.currentPos());
	}

	static macro function names(...exprs:haxe.macro.Expr):haxe.macro.Expr {
		var strings = exprs.toArray().map(e -> macro $v{haxe.macro.ExprTools.toString(e)});
		return macro $a{strings};
	}
}
#end

class Builder {
	public static function build() {
		var fields = haxe.macro.Context.getBuildFields();
		fields.push((macro class X {
			public static var tag = "built";
		}).fields[0]);
		return fields;
	}
}
