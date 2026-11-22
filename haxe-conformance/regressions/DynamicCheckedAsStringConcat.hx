// `(d : String)` of a Dynamic holding a String concatenates as that String.
class DynamicCheckedAsStringConcat {
	static function optional(?from:Dynamic):String
		return "o:" + (from : String);

	static function required(from:Dynamic):String
		return "r:" + (from : String);

	static function viaLocal(from:Dynamic):String {
		var s:String = (from : String);
		return "l:" + s + ":" + s.length;
	}

	static function main() {
		var d:Dynamic = "word";
		var got = [optional(d), required(d), viaLocal(d), (d : String) + "!"].join(" ");
		if (got != "o:word r:word l:word:4 word!")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
