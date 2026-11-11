typedef Rec = {
	final a:Int;
	final b:Int;
}

@:build(AbstractBuildMac.tag("Plain"))
abstract Plain(Int) {
	public inline function new(v:Int)
		this = v;
}

@:build(AbstractBuildMac.tag("Tagged"))
enum abstract Tagged(Int) {
	var One = 1;
}

@:build(AbstractBuildMac.keys("AbstractBuildMacros.Rec"))
enum abstract Key(String) {
	var A = "a";
	var B = "b";
}

class AbstractBuildMacros {
	static function main() {
		var r:Rec = {a: 10, b: 20};
		var got = [Plain.TAG_Plain, Tagged.TAG_Tagged, "" + Key.A.of(r), "" + Key.B.of(r)].join(",");
		if (got != "Plain,Tagged,10,20")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
