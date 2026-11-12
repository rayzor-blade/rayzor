// An optional Int, Float or Bool field of a structure is Null<T>: a zero
// written in a literal is a value, an omitted field is null.
typedef OptionalFieldZeroAttr = {
	?n:Int,
	?f:Float,
	?b:Bool,
	?s:String
}

class OptionalFieldZeroValue {
	static function show(a:OptionalFieldZeroAttr):String {
		return (a.n == null ? "null" : "" + a.n) + " " + (a.f == null ? "null" : "" + a.f) + " "
			+ (a.b == null ? "null" : "" + a.b);
	}

	static function main() {
		var wrapOff:OptionalFieldZeroAttr = {b: false};
		var wrap = wrapOff.b != null ? wrapOff.b : true;
		var got = show({n: 0, f: 0.0, b: false}) + "|" + show({n: 3, f: 1.5, b: true}) + "|" + show({}) + "|" + wrap;
		if (got != "0 0 false|3 1.5 true|null null null|false")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
