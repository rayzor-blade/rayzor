package pk;

import haxe.macro.Expr;

class M {
	public static macro function outer(e:Expr):Expr
		return macro [for (i in 0...3) pk.M.inner(i)];

	public static macro function inner(e:Expr):Expr
		return macro $e * 10;

	public static macro function withPos():Expr {
		var e = macro 5;
		return macro @:pos(e.pos) pk.M.inner(5);
	}

	public static macro function direct():Expr
		return macro pk.M.inner(4);
}
