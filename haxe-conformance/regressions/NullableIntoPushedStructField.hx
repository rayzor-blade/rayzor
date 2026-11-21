// A `Null<Int>` stored into an `Int` field of an object literal pushed onto an
// array of that structure is the Int it holds.
typedef Origin = {name:String, line:Int};

class NullableIntoPushedStructField {
	static function main() {
		var out:Array<Origin> = [];
		var parts = "a|42".split("|");
		out.push({name: parts[0], line: Std.parseInt(parts[1])});
		out.unshift({name: "b", line: Std.parseInt("7")});
		var got = out.map(o -> o.name + o.line).join(",");
		if (got != "b7,a42")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
