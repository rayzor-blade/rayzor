// An untyped `[]` passed for an Array<String> parameter is an Array<String>.
class Filler {
	public function new() {}

	public function add(a:Array<String>):Void
		a.push("m");
}

class EmptyArrayFromArgument {
	static function fill(a:Array<String>):Void {
		a.push("one");
		a.push("two " + a.length);
	}

	static function main() {
		var out = [];
		fill(out);
		var viaMethod = [];
		new Filler().add(viaMethod);
		var handlers = new Map<String, (Array<String>, String) -> Void>();
		handlers.set("h", (o, v) -> o.push(v + "!"));
		var viaValue = [];
		handlers.get("h")(viaValue, "x");
		var local:(Array<String>) -> Void = a -> a.push("l");
		var viaLocal = [];
		local(viaLocal);
		var got = [out.join(","), out[0], viaMethod[0], viaValue.join(""), viaLocal[0]].join("|");
		if (got != "one,two 1|one|m|x!|l")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
