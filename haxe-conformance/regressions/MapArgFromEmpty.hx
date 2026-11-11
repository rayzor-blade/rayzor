// `new Map()` and `[]` passed for a Map parameter build the parameter's map.
typedef Decl = {value:String, line:Int};

class MapArgFromEmpty {
	static function count(name:String, from:Map<String, Decl>, ?tag:String):String
		return name + Lambda.count(from) + (tag == null ? "" : tag);

	static function ints(m:Map<Int, String>):Int {
		m.set(1, "a");
		return Lambda.count(m);
	}

	static function main() {
		var got = [count("a", new Map(), "!"), count("b", new Map()), count("c", []), "" + ints(new Map()), "" + ints([])].join(",");
		if (got != "a0!,b0,c0,1,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
