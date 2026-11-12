// An arrow function's block returns its last value even beside early returns.
class ArrowEarlyReturnTail {
	static function main() {
		var f:(Int) -> Array<Int> = n -> {
			if (n == 0)
				return [];
			[n + 100];
		};
		var g = (n:Int) -> {
			if (n == 0)
				return 5;
			n * 2;
		};
		var h = new Map<String, (String, Int) -> Array<Int>>();
		h.set("x", (v, n) -> {
			if (v == "none")
				return [0];
			var out:Array<Int> = [n];
			out;
		});
		var got = f(7) + "," + f(0) + "," + g(3) + "," + g(0) + "," + h["x"]("u", 4) + "," + h["x"]("none", 4);
		if (got != "[107],[],6,5,[4],[0]")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
