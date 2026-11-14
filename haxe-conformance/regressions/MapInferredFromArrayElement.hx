// `new Map()` pushed into an array of maps takes the element's type arguments.
private class Pools {
	final pools:Array<Map<String, Array<Int>>> = [];

	public function new() {}

	public function add():Map<String, Array<Int>> {
		pools.push(new Map());
		return pools[pools.length - 1];
	}
}

class MapInferredFromArrayElement {
	static function main() {
		var l:Array<Map<String, Int>> = [];
		l.push(new Map());
		l[0].set("a", 1);
		var ints:Array<Map<Int, String>> = [];
		ints.unshift(new Map());
		ints.insert(1, new Map());
		ints[0].set(3, "x");
		ints[1].set(4, "y");
		var pool = new Pools().add();
		pool.set("k", [7]);
		var got = l[0].get("a") + " " + ints[0].get(3) + ints[1].get(4) + " " + pool.get("k")[0];
		if (got != "1 xy 7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
