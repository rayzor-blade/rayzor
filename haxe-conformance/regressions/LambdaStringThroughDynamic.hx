// A lambda passed for a `Void->Dynamic` returns its String boxed, as it does
// an Int, Float or Bool.
class LambdaStringThroughDynamic {
	static function show(f:Void->Dynamic):String
		return "" + f();

	static function main() {
		var s = "v";
		var got = [
			show(() -> "yes"), show(() -> s), show(() -> {
				var q = "b";
				q;
			}), show(() -> 3), show(() -> 1.5), show(() -> true)
		].join(",");
		if (got != "yes,v,b,3,1.5,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
