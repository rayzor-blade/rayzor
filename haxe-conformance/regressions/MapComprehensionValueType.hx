typedef T = {id:String, tags:Array<String>, style:Map<String, String>};

// A map comprehension's values keep their type: an Array or a structure
// read back from it is used as one.
class MapComprehensionValueType {
	static function main() {
		var words = ["alpha", "beta", "gamma"];
		var n = [for (w in words) w.length => [w, w + "!"]];
		var a:Array<T> = [
			{id: "a", tags: ["x"], style: ["color" => "red"]},
			{id: "b", tags: ["y", "z"], style: ["color" => "blue"]}
		];
		var m = [for (e in a) e.id => e];
		var lengths = [for (w in words) w => w.length];
		var got = [
			n.get(5).join("+"),
			m.get("b").tags.join(","),
			m.get("a").style.get("color"),
			Std.string(lengths.get("beta") + 1)
		].join(" ");
		if (got != "gamma+gamma! y,z red 5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
