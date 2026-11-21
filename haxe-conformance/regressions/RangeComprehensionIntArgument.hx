// A range comprehension's variable is an Int wherever it goes, including a
// Float parameter of a runtime Math function.
class RangeComprehensionIntArgument {
	static final SCALE = [for (e in 0...4) Math.pow(2, e - 2)];

	static function main() {
		var roots = [for (k in 0...3) Math.sqrt(k)];
		var maxes = [for (k in 0...3) Math.max(k, 0.5)];
		var got = SCALE.join(",") + " " + roots[1] + " " + Math.round(roots[2] * 1000) + " " + maxes.join(",");
		if (got != "0.25,0.5,1,2 1 1414 0.5,1,2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
