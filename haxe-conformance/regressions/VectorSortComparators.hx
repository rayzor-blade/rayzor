// Vector.sort orders by the comparator in O(n log n) comparisons, and a
// comparator that is not a consistent order still leaves every element.
import haxe.ds.Vector;

class VectorSortComparators {
	static function main() {
		var n = 4096;
		var calls = 0;
		var ints = new Vector<Int>(n);
		for (i in 0...n)
			ints[i] = (i * 7919) % n;
		ints.sort((a, b) -> {
			calls++;
			a - b;
		});
		var ordered = true;
		for (i in 0...n)
			if (ints[i] != i)
				ordered = false;

		var floats = new Vector<Float>(5);
		for (i in 0...5)
			floats[i] = [0.5, -1.25, 3.0, 0.5, -7.5][i];
		floats.sort((a, b) -> a < b ? -1 : (a > b ? 1 : 0));

		var names = new Vector<String>(4);
		for (i in 0...4)
			names[i] = ["pear", "fig", "apple", "kiwi"][i];
		names.sort((a, b) -> a < b ? -1 : (a > b ? 1 : 0));

		var mixed = new Vector<Int>(n);
		var sum = 0;
		for (i in 0...n) {
			mixed[i] = (i * 31) % 97;
			sum += mixed[i];
		}
		mixed.sort((a, b) -> (a * 7 + b * 3) % 5 - 2);
		var kept = 0;
		for (v in mixed)
			kept += v;

		var got = ordered + " " + (calls < n * 24) + " " + floats.join(",") + " " + names.join(",") + " " + (kept == sum);
		if (got != "true true -7.5,-1.25,0.5,0.5,3 apple,fig,kiwi,pear true")
			throw got + " calls=" + calls;
		Sys.println("CONFORMANCE_OK");
	}
}
