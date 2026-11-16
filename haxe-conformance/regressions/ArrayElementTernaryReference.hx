// A ternary between an element read from an array of arrays and an array
// local yields the array, not a box of its address.
class ArrayElementTernaryReference {
	static function main() {
		var q:Array<Array<Float>> = [[1.0, 2.0, 3.0]];
		var none:Array<Float> = [0.0, 0, 0];
		var out = [];
		for (i in 0...2) {
			var x = i < q.length ? q[i] : none;
			out.push(x.length + ":" + x[2]);
		}
		var names:Array<Array<String>> = [["a", "b"]];
		var empty:Array<String> = [];
		var n = names.length > 0 ? names[0] : empty;
		out.push(n.join("|"));
		var got = out.join(",");
		if (got != "3:3,3:0,a|b")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
