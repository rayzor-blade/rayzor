import haxe.macro.Expr;

class DivMac {
	public static macro function ratios():Expr {
		var parsed = Std.parseInt("20");
		var typed:Float = parsed / 100;
		var fromLiterals = 7 / 2;
		var wrapped = 9 % 4;
		return macro $v{parsed / 100} + "," + $v{typed} + "," + $v{fromLiterals} + "," + $v{wrapped};
	}
}
