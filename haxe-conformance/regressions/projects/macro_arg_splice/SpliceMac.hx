import haxe.macro.Expr;

class SpliceMac {
	public static macro function beside():Expr {
		var list = [macro 10, macro 20, macro 30];
		return macro MacroArgSplice.sum(1, $a{list});
	}

	public static macro function alone():Expr {
		var list = [macro 1, macro 2, macro 3];
		return macro MacroArgSplice.three($a{list});
	}

	public static macro function objects():Expr {
		var stops = [for (i in 0...3) macro {offset: $v{i * 0.5}, alpha: 1.0}];
		return macro MacroArgSplice.count(true, $a{stops});
	}
}
