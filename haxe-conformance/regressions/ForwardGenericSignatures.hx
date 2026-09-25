// A generic static declared BELOW its caller: the forward signature keeps the
// method's own type parameters, so its result still binds the caller's
// generic call.
private class Checks {
	public function new() {}

	public function same<T>(a:Array<T>, b:Array<T>):Bool
		return a.length == b.length && Std.string(a[0]) == Std.string(b[0]);
}

class ForwardGenericSignatures extends Checks {
	function run():Bool {
		var words = Later.wrap('hello');
		var numbers = Later.wrap(7);
		return same(['hello'], words) && same([7], numbers) && Later.first(numbers) == 7;
	}

	static function main() {
		if (new ForwardGenericSignatures().run())
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL");
	}
}

private class Later {
	public static function wrap<T>(v:T):Array<T> return [v];

	public static function first<S>(a:Array<S>):S return a[0];
}
