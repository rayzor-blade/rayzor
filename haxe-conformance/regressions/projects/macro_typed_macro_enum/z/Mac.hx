package z;

import haxe.macro.Expr;

class Mac {
	public static macro function bare(arg:Expr):Expr {
		var ct:ComplexType = macro :Int;
		var path:TypePath = {pack: ["z"], name: "Box", params: [TPType(ct)]};
		var target:ComplexType = TPath({pack: ["z"], name: "Box", params: [TPType(ct)]});
		return macro (cast new $path(cast $arg) : $target);
	}

	public static macro function qualified(arg:Expr):Expr {
		var ct:ComplexType = macro :Int;
		var path:TypePath = {pack: ["z"], name: "Box", params: [TPType(ct)]};
		var target:ComplexType = ComplexType.TPath({pack: ["z"], name: "Box", params: [TPType(ct)]});
		return macro (cast new $path(cast $arg) : $target);
	}

	public static macro function anon():Expr {
		var members:Array<haxe.macro.Expr.Field> = [for (f in ["a", "b"]) {name: f, kind: FVar(macro :Int), pos: haxe.macro.Context.currentPos()}];
		var qualified = haxe.macro.Expr.ComplexType.TAnonymous(members);
		var short = ComplexType.TAnonymous(members);
		var bare = TAnonymous(members);
		var p = new haxe.macro.Printer();
		return macro $v{[for (ct in [qualified, short, bare]) p.printExpr(macro var x:$ct)].join("|")};
	}

	public static macro function names():Expr {
		var ct:ComplexType = macro :Int;
		var typed:ComplexType = TPath({pack: ["z"], name: "Box", params: [TPType(ct)]});
		var qual = ComplexType.TPath({pack: ["z"], name: "Box", params: [TPType(ct)]});
		var info = Type.getEnumName(Type.getEnum(typed)) + ":" + Type.enumConstructor(typed) + " "
			+ Type.getEnumName(Type.getEnum(qual)) + ":" + Type.enumConstructor(qual);
		return macro $v{info};
	}
}
