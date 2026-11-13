// A method called on a Null<String> keeps String's return type, so a
// further method call on the result dispatches as String.
class NullableStringMethodResult {
	static function resolved(name:String):Null<String>
		return name == "none" ? null : "Thumb CurrentColor";

	static function has(name:String):String {
		var colors = resolved(name);
		return colors != null && colors.toLowerCase().indexOf("currentcolor") >= 0 ? "has" : "no";
	}

	static function at(name:String):Int {
		var colors = resolved(name);
		if (colors == null)
			return -2;
		return colors.toLowerCase().indexOf("currentcolor");
	}

	static function main() {
		var got = [has("x"), has("none"), "" + at("x"), "" + at("none"), "" + resolved("x").toUpperCase().indexOf("C")].join(",");
		if (got != "has,no,6,-2,6")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
