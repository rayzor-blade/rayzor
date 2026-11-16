// A specialized body passes a key it read from a node to a method dispatched
// through the vtable, whose key parameter stays an erased slot.
class DispatchedErasedKeys {
	static function main() {
		var t = new haxe.ds.BalancedTree<Int, Bool>();
		for (k in [5, -3, 12, 0, 7, -40, 100, 2])
			t.set(k, k % 2 == 0);
		var order = [for (k in t.keys()) k].join(",");
		t.set(7, true);
		var found = t.exists(7) + " " + t.get(7) + " " + t.get(12) + " " + t.remove(-40);
		var got = order + "|" + found + "|" + [for (k in t.keys()) k].length;
		if (got != "-40,-3,0,2,5,7,12,100|true true true true|7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
