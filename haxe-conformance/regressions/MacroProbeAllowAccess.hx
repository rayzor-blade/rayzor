import haxe.macro.Context;
import haxe.macro.Expr;

// A macro typing an expression honours `@:allow` on the member or its class,
// and `@:access` on the class doing the access.
class AllowOwner {
	@:allow(AllowFriend)
	static function matches(x:Int):Bool
		return x > 1;

	@:allow(AllowFriend)
	var secret = 7;

	static var hidden = 3;

	public function new() {}
}

@:allow(AllowFriend)
class AllowWholeClass {
	static var count = 4;
}

class AllowFriend {
	public static function run(owner:AllowOwner):String {
		var viaMacro = MacroProbeAllowAccess.typed(() -> AllowOwner.matches(3));
		var field = MacroProbeAllowAccess.typed(() -> owner.secret);
		var whole = MacroProbeAllowAccess.typed(() -> AllowWholeClass.count);
		var denied = MacroProbeAllowAccess.errorText(AllowOwner.hidden);
		return viaMacro + " " + field + " " + whole + " " + denied;
	}
}

@:access(AllowOwner)
class AccessGranted {
	public static function run():Int
		return MacroProbeAllowAccess.typed(() -> AllowOwner.hidden);
}

class MacroProbeAllowAccess {
	public macro static function typed(fn:Expr):Expr {
		Context.typeof(macro $fn());
		return macro $fn();
	}

	public macro static function errorText(expression:Expr):Expr {
		var message = try {
			Context.typeof(expression);
			null;
		} catch (error:haxe.macro.Expr.Error) {
			error.message;
		};
		return macro $v{message};
	}

	#if !macro
	static function main() {
		var got = AllowFriend.run(new AllowOwner()) + " " + AccessGranted.run();
		if (got != "true 7 4 Cannot access private field hidden 3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
	#end
}
