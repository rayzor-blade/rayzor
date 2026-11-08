class NewThroughNullableHint {
	static var registry = new Map<Int, Map<String, Int>>();

	static function register(key:Int, name:String, value:Int):Void {
		var nodes = registry.get(key);
		if (nodes == null) {
			registry.set(key, nodes = new Map());
		}
		nodes.set(name, value);
	}

	static function main() {
		var m = new Map<Int, Map<String, Int>>();
		var nodes = m.get(1);
		nodes = new Map();
		nodes.set("a", 1);
		if (nodes.get("a") != 1) throw "assigned map";

		var again = m.get(2);
		if (again == null) m.set(2, again = new Map());
		again.set("b", 2);
		if (m.get(2).get("b") != 2) throw "map stored through assignment";

		register(7, "x", 5);
		register(7, "y", 6);
		if (registry.get(7).get("x") != 5 || registry.get(7).get("y") != 6) throw "register";
		Sys.println("CONFORMANCE_OK");
	}
}
