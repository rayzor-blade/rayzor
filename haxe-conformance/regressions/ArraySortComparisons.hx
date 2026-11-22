// Array.sort orders by the comparator in O(n log n) comparisons.
class ArraySortComparisons {
	static function main() {
		var n = 4096;
		var calls = 0;
		var ints = [for (i in 0...n) (i * 7919) % n];
		ints.sort((a, b) -> {
			calls++;
			a - b;
		});
		var ordered = true;
		for (i in 0...n)
			if (ints[i] != i)
				ordered = false;

		var items = [for (i in 0...n) {key: (i * 31) % 7, at: i}];
		items.sort((a, b) -> a.key - b.key);
		var grouped = true;
		for (i in 1...n)
			if (items[i - 1].key > items[i].key)
				grouped = false;

		var floats = [0.5, -1.25, 3.0, 0.5, -7.5];
		floats.sort((a, b) -> a < b ? -1 : (a > b ? 1 : 0));

		var got = ordered + " " + grouped + " " + (calls < n * 24) + " " + floats.join(",");
		if (got != "true true true -7.5,-1.25,0.5,0.5,3")
			throw got + " calls=" + calls;
		Sys.println("CONFORMANCE_OK");
	}
}
