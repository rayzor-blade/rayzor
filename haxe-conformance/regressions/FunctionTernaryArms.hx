// `cond ? lambda : staticMethod` joins two closures; neither arm is boxed.
class FunctionTernaryArms {
	static function compact(v:Float):String
		return "c" + v;

	static function main() {
		var got = [];
		for (log in [false, true]) {
			var f:Float->String = log ? v -> "x" + v : compact;
			var g:Float->String = log ? compact : v -> "y" + v;
			got.push(f(1) + g(2));
		}
		if (got.join(",") != "c1y2,x1c2")
			throw got.join(",");
		Sys.println("CONFORMANCE_OK");
	}
}
