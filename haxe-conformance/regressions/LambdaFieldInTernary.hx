typedef Props = {format:Float->String};

// A lambda arm of a conditional in a structure field takes the field's
// parameter types: `v` is a Float, not inferred from `"y" + v`.
class LambdaFieldInTernary {
	static function compact(v:Float):String
		return "c" + v;

	static function main() {
		var got = [];
		for (log in [false, true]) {
			var p:Props = {format: log ? v -> "y" + (v * 2) : LambdaFieldInTernary.compact};
			got.push(p.format(1.5));
		}
		if (got.join(",") != "c1.5,y3")
			throw got.join(",");
		Sys.println("CONFORMANCE_OK");
	}
}
