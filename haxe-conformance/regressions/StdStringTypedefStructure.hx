// Std.string of a structure typed by a typedef prints its fields, whether it
// is a local or an array element.
typedef StdStringTypedefStructureChange = {
	var originalStart:Int;
	var originalEnd:Int;
}

class StdStringTypedefStructure {
	static function make():StdStringTypedefStructureChange
		return {originalStart: 1, originalEnd: 3};

	static function main() {
		var c = make();
		var out = [Std.string(c)];
		var arr:Array<StdStringTypedefStructureChange> = [make()];
		for (x in arr)
			out.push("change " + Std.string(x));
		var got = out.join(" | ");
		if (got != "{originalEnd: 3, originalStart: 1} | change {originalEnd: 3, originalStart: 1}"
			&& got != "{originalStart: 1, originalEnd: 3} | change {originalStart: 1, originalEnd: 3}")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
