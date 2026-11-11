class Names {
	static final byName:Map<String, Int> = ["a" => 1, "div" => 2];
	static final flags:Map<String, Bool> = [for (n in ["p", "span"]) n => true];
	static final byId:Map<Int, String> = [7 => "seven"];
	static var later:Map<String, String>;

	public static function report():String {
		later = ["k" => "v"];
		var pairs = [for (k => v in byName) k + v];
		pairs.sort(Reflect.compare);
		return [
			"" + byName.exists("div"), "" + byName.get("a"), "" + (flags.get("span") == true), byId.get(7), later.get("k"),
			pairs.join("+")
		].join(",");
	}
}
