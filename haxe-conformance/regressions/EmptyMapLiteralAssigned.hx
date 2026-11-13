// `[]` where a Map is expected is an empty map, in a field initializer and in
// a plain assignment as in a declaration.
class EmptyMapHolder {
	var hidden:Map<String, Bool> = [];
	final fixed:Map<Int, String> = [];

	public function new() {}

	public function reset():Void
		hidden = [];

	public function put(k:String):Int {
		hidden.set(k, true);
		fixed.set(k.length, k);
		return [for (_ in hidden.keys()) 1].length;
	}
}

class EmptyMapLiteralAssigned {
	static function main() {
		var h = new EmptyMapHolder();
		var a = h.put("x") + h.put("y");
		h.reset();
		var b = h.put("z");
		var m:Map<String, Int> = new Map();
		m.set("old", 1);
		m = [];
		m.set("q", 2);
		var got = a + " " + b + " " + m.get("q") + " " + m.exists("old");
		if (got != "3 1 2 false")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
