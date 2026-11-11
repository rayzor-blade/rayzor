// A Null<Map> iterates as the map it holds: keys, values and key => value.
class Holder {
	public var m:Null<Map<String, String>> = null;
	public var list:Null<Array<Int>> = null;

	public function new() {}
}

class NullableMapIteration {
	static function main() {
		var holder = new Holder();
		holder.m = new Map();
		holder.m.set("a", "1");
		holder.list = [4, 5];
		var nullable:Null<Map<String, String>> = holder.m;
		var s = "";
		for (k => v in nullable)
			s += k + v;
		var total = 0;
		for (x in holder.list)
			total += x;
		var got = [
			[for (k => v in nullable) k + v].join(","), [for (v in nullable) v].join(","), s,
			[for (k => v in holder.m) k + "=" + v].join(","), "" + total
		].join("|");
		if (got != "a1|1|a1|a=1|9")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
