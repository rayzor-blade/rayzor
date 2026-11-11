import haxe.macro.Expr;
import Own.Shape;

using Own.OwnExt;

class MacUser {
	// a user enum, an extension called on a constructor application
	public static macro function onCtor():Expr
		return macro $v{Shape.Circle(3).describe()};

	// a user enum in a variable
	public static macro function onVar():Expr {
		var s = Shape.Circle(4);
		return macro $v{s.describe()};
	}

	// with an argument
	public static macro function withArg():Expr {
		var s = Shape.Square;
		return macro $v{s.twice(2)};
	}
}
