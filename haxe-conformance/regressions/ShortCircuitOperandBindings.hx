// A variable that only the right operand of `&&`, `||` or `??` reads or
// assigns keeps a value on the edge that skips that operand.
class ShortCircuitOperandBindings {
	static function run(b:Array<Int>, n:Null<Int>):String {
		var at = 0;
		function next():Int
			return b[at++];
		var x = 1;
		var y = 1;
		var z = 1;
		var hits = 0;
		for (k in 0...3) {
			if (b[at] == 2 && b[at + 1] == 2) {
				at += 2;
				var c = 0;
				while (c < 2) {
					hits += next();
					c++;
				}
			} else
				at++;
			if (k > 0 && (x = x + 10) > 0) {}
			if (k == 0 || (y = y * 2) > 0) {}
			z = (n ?? (z = z + 100)) + z;
		}
		return '$at $hits $x $y $z';
	}

	static function main() {
		var got = "";
		for (_ in 0...2000)
			got = run([2, 2, 5, 6, 1, 2, 2, 7, 8], null) + " " + run([1, 1, 1], 3);
		if (got != "9 26 21 4 1408 3 0 21 4 10")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
