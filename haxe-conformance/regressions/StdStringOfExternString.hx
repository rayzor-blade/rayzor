// Std.string of a String an extern returns untyped is that String.
class StdStringOfExternString {
	static function main() {
		var x = "ab";
		var s:String = [x, x].join("");
		var d:Dynamic = [x, x].join("-");
		var got = Std.string([x, x].join("")) + "|" + Std.string(s) + "|" + Std.string(["a", "b"].join(",")) + "|" + Std.string(d);
		if (got != "abab|abab|a,b|ab-ab")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
