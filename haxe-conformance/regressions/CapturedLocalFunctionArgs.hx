// A local function called from a lambda types its arguments by its own
// parameters: a structure literal for a `@:structInit` class builds one.
@:structInit
class CapturedLocalFunctionNotch {
	public final a:Float;
	public final b:Float;

	public function new(a:Float = 0, b:Float = 2) {
		this.a = a;
		this.b = b;
	}
}

class CapturedLocalFunctionArgs {
	static function main() {
		function shape(n:CapturedLocalFunctionNotch, w:Float, h:Float, bordered:Bool):String
			return n.a + " " + w + " " + h + " " + bordered;
		var direct = shape({a: 1}, 10, 20, true);
		var viaLambda = () -> shape({a: 4}, 160, 70, false);
		var nested = () -> (() -> shape({a: 5}, 1, 2, true))();
		var got = direct + "|" + viaLambda() + "|" + nested();
		if (got != "1 10 20 true|4 160 70 false|5 1 2 true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
