// A generic Map<String, T>.get of a stored zero is the value, not null.
enum abstract GenericMapZeroDir(Int) to Int {
	var Row = 0;
	var Column = 1;
}

class GenericMapZeroValue {
	static function look<T>(values:Map<String, T>, k:String):String {
		var x = values.get(k);
		if (x == null)
			return "null";
		return "found " + Std.string(x);
	}

	static function main() {
		var dirs = ["row" => GenericMapZeroDir.Row, "column" => GenericMapZeroDir.Column];
		var ints = ["zero" => 0, "one" => 1];
		var floats = ["zero" => 0.0];
		var got = [look(dirs, "row"), look(dirs, "none"), look(ints, "zero"), look(ints, "one"), look(floats, "zero")].join(",");
		if (got != "found 0,null,found 0,found 1,found 0")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
